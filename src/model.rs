use anyhow::Result;
use candle_core::{DType, Device, Tensor, D, IndexOp};
use candle_nn::{ops, loss, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
use serde::{Deserialize, Serialize};

pub const IMAGE_DIM: usize = 784;
pub const LABELS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    pub weight: Vec<f32>, // flattened [out * in]
    pub bias: Vec<f32>,   // [out]
    pub out_dim: usize,
    pub in_dim: usize,
}

impl ModelParams {
    pub fn zeros_linear() -> Self {
        Self { weight: vec![0.0; LABELS * IMAGE_DIM], bias: vec![0.0; LABELS], out_dim: LABELS, in_dim: IMAGE_DIM }
    }
}

pub fn random_linear_params(dev: &Device) -> Result<ModelParams> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let lin = linear_z(IMAGE_DIM, LABELS, vs)?;
    let w = lin.weight().reshape((LABELS * IMAGE_DIM,))?.to_vec1::<f32>()?;
    let b = lin.bias().unwrap().to_vec1::<f32>()?;
    Ok(ModelParams { weight: w, bias: b, out_dim: LABELS, in_dim: IMAGE_DIM })
}

pub fn linear_from_params(params: &ModelParams, dev: &Device) -> Result<Linear> {
    let w = Tensor::from_vec(params.weight.clone(), (params.out_dim, params.in_dim), dev)?;
    let b = Tensor::from_vec(params.bias.clone(), params.out_dim, dev)?;
    Ok(Linear::new(w, Some(b)))
}

pub fn extract_params_from_linear(lin: &Linear) -> Result<ModelParams> {
    let w = lin.weight().reshape((LABELS * IMAGE_DIM,))?.to_vec1::<f32>()?;
    let b = lin.bias().unwrap().to_vec1::<f32>()?;
    Ok(ModelParams { weight: w, bias: b, out_dim: LABELS, in_dim: IMAGE_DIM })
}

pub fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> anyhow::Result<Linear> {
    use candle_nn::init;
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", init::DEFAULT_KAIMING_NORMAL)?;
    let bs = vs.get_with_hints(out_dim, "bias", init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

pub struct LocalTrainStats {
    pub final_loss: f32,
    pub test_accuracy: f32,
}

pub fn local_train_linear(
    mut params: ModelParams,
    epochs: usize,
    learning_rate: f64,
    shard_index: usize,
    total_shards: usize,
    dev: &Device,
) -> anyhow::Result<(ModelParams, LocalTrainStats)> {
    // Load MNIST on demand (CPU ok)
    let m = candle_datasets::vision::mnist::load()?;

    // Partition dataset by contiguous chunks per shard
    let train_n = m.train_images.dim(0)?;
    let test_n = m.test_images.dim(0)?;
    let shard_train = shard_range(train_n, shard_index, total_shards);
    let shard_test = shard_range(test_n, shard_index, total_shards);

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(dev)?;
    let train_images = m.train_images.to_device(dev)?;
    let test_images = m.test_images.to_device(dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(dev)?;

    // Build model from params managed by a VarMap for optim
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    // Build trainable model via VarBuilder and align its params to provided `params`
    let mut train_linear = super_make_linear_vs(vs.clone())?;
    // Pre-align: minimize MSE between variables and provided params for a few steps
    {
        let target_w = Tensor::from_vec(params.weight.clone(), (LABELS, IMAGE_DIM), dev)?;
        let target_b = Tensor::from_vec(params.bias.clone(), LABELS, dev)?;
        let mut align_opt = SGD::new(varmap.all_vars(), 1.0)?;
        for _ in 0..3 {
            let dw = (train_linear.weight() - &target_w)?.sqr()?.sum_all()?;
            let db = (train_linear.bias().unwrap() - &target_b)?.sqr()?.sum_all()?;
            let align_loss = (&dw + &db)?;
            align_opt.backward_step(&align_loss)?;
        }
    }

    let mut sgd = SGD::new(varmap.all_vars(), learning_rate)?;

    let mut last_loss = 0.0f32;
    for _epoch in 0..epochs {
        let xs = train_images.i(shard_train.clone())?;
        let ys = train_labels.i(shard_train.clone())?;
        let logits = train_linear.forward(&xs)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &ys)?;
        sgd.backward_step(&loss)?;
        last_loss = loss.to_scalar::<f32>()?;
    }
    // Evaluate on this shard's test split
    let test_logits = train_linear.forward(&test_images.i(shard_test.clone())?)?;
    let sum_ok = test_logits
        .argmax(D::Minus1)?
        .eq(&test_labels.i(shard_test)?)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let test_accuracy = sum_ok / (test_n / total_shards) as f32;

    let new_params = extract_params_from_linear(&train_linear)?;
    Ok((new_params, LocalTrainStats { final_loss: last_loss, test_accuracy }))
}

fn super_make_linear_vs(vs: VarBuilder) -> anyhow::Result<Linear> {
    let w = vs.get_with_hints((LABELS, IMAGE_DIM), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
    let b = vs.get_with_hints(LABELS, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(w, Some(b)))
}

fn shard_range(n: usize, shard_index: usize, total_shards: usize) -> std::ops::Range<usize> {
    let base = n / total_shards;
    let rem = n % total_shards;
    let start = shard_index * base + rem.min(shard_index);
    let len = base + if shard_index < rem { 1 } else { 0 };
    start..(start + len)
}