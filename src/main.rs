mod model;
mod types;
mod server;
mod client;

use clap::{Parser, Subcommand};
use tracing_subscriber::{fmt, EnvFilter};

#[derive(Parser)]
#[command(version, about = "FedAvg with Candle: parameter server and clients", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Original sequential demo on a single process
    Demo {
        #[arg(long, default_value_t = 1.0)]
        learning_rate: f64,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
    },
    // Start parameter server
    Server {
        #[arg(long, default_value = "127.0.0.1:3000")]
        bind: String,
    },
    // Start federated client and optionally register to server
    Client {
        #[arg(long, default_value = "127.0.0.1:4000")]
        bind: String,
        #[arg(long)]
        server: Option<String>,
        #[arg(long, default_value = "linear")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();
    match Cli::parse().command {
        Commands::Server { bind } => server::run(bind).await,
        Commands::Client { bind, server, model } => client::run(bind, server, model).await,
        Commands::Demo { learning_rate, epochs } => demo(learning_rate, epochs),
    }
}

// Original demo training: trains two local models on split mnist, then averages
fn demo(learning_rate: f64, epochs: usize) -> anyhow::Result<()> {
    use candle_core::{DType, Tensor, D, IndexOp};
    use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};

    const IMAGE_DIM: usize = model::IMAGE_DIM;
    const LABELS: usize = model::LABELS;

    struct LinearModel { linear: Linear }
    trait Model: Sized {
        fn new(vs: VarBuilder) -> candle_core::Result<Self>;
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor>;
        fn weight(&self) -> candle_core::Result<&Tensor>;
        fn bias(&self) -> candle_core::Result<&Tensor>;
    }
    fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> candle_core::Result<Linear> {
        let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
        let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
        Ok(Linear::new(ws, Some(bs)))
    }
    impl Model for LinearModel {
        fn new(vs: VarBuilder) -> candle_core::Result<Self> {
            let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
            Ok(Self { linear })
        }
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> { self.linear.forward(xs) }
        fn weight(&self) -> candle_core::Result<&Tensor> { Ok(self.linear.weight()) }
        fn bias(&self) -> candle_core::Result<&Tensor> { Ok(self.linear.bias().unwrap()) }
    }

    fn model_train<M: Model>(model: M, test_images: &Tensor, test_labels: &Tensor, train_images: &Tensor, train_labels: &Tensor, mut sgd: SGD, epochs: usize) -> candle_core::Result<M> {
        for epoch in 1..epochs+1 {
            let logits = model.forward(&train_images)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;

            let test_logits = model.forward(&test_images)?;
            let sum_ok = test_logits.argmax(D::Minus1)?.eq(test_labels)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!("{epoch:4} train loss: {:8.5} test acc: {:5.2}%", loss.to_scalar::<f32>()?, 100. * test_accuracy);
        }
        Ok(model)
    }

    let dev = candle_core::Device::cuda_if_available(0).unwrap_or_else(|_| candle_core::Device::Cpu);
    let m = candle_datasets::vision::mnist::load()?;

    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let train_images = m.train_images.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model_1 = model_train(LinearModel::new(vs.clone())?, &test_images.i(..test_images.shape().dims()[0]/2)?, &test_labels.i(..test_labels.shape().dims()[0]/2)?, &train_images.i(..train_images.shape().dims()[0]/2)?, &train_labels.i(..train_labels.shape().dims()[0]/2)?, SGD::new(varmap.all_vars(), learning_rate)?, epochs)?;

    let varmap2 = VarMap::new();
    let vs2 = VarBuilder::from_varmap(&varmap2, DType::F32, &dev);
    let model_2 = model_train(LinearModel::new(vs2.clone())?, &test_images.i(test_images.shape().dims()[0]/2..)?, &test_labels.i(test_labels.shape().dims()[0]/2..)?, &train_images.i(train_images.shape().dims()[0]/2..)?, &train_labels.i(train_labels.shape().dims()[0]/2..)?, SGD::new(varmap2.all_vars(), learning_rate)?, epochs)?;

    let model = Linear::new(((model_1.weight()?+ model_2.weight()?)?/2.0)?, Some(((model_1.bias().unwrap()+ model_2.bias().unwrap())?/2.0)?));
    let test_logits = model.forward(&test_images)?;
    let sum_ok = test_logits.argmax(D::Minus1)?.eq(&test_labels)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    let test_accuracy = sum_ok / test_labels.dims1()? as f32;
    println!("Average test accuracy: {:5.2}%", 100. * test_accuracy);

    Ok(())
}
