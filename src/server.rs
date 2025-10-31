use std::{collections::HashMap, sync::Arc};

use axum::{extract::State, routing::{get, post}, Json, Router};
use parking_lot::RwLock;
use serde::Deserialize;
use tracing::{error, info};
use uuid::Uuid;
use tokio::net::TcpListener;
use candle_nn::Module;

use crate::model::{linear_from_params, random_linear_params, IMAGE_DIM, LABELS, ModelParams};
use crate::types::*;

#[derive(Clone)]
pub struct ServerState {
    inner: Arc<RwLock<InnerState>>,
}

struct InnerState {
    models: HashMap<String, GlobalModel>,
}

struct GlobalModel {
    status: ModelStatus,
    params: Option<ModelParams>,
    clients: Vec<ClientInfo>,
}

#[derive(Clone)]
struct ClientInfo {
    id: Uuid,
    url: String,
    shard: usize,
    total_shards: usize,
}

impl ServerState {
    pub fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(InnerState { models: HashMap::new() })) }
    }
}

pub async fn run(bind: String) -> anyhow::Result<()> {
    let state = ServerState::new();

    let app = Router::new()
        .route("/register", post(register))
        .route("/init", post(init_model))
        .route("/train", post(train))
        .route("/get", get(get_model))
        .route("/test", post(test_model))
        .with_state(state);

    info!("Parameter server listening on {}", bind);
    let listener = TcpListener::bind(&bind).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn register(State(state): State<ServerState>, Json(req): Json<RegisterRequest>) -> Json<RegisterResponse> {
    let mut g = state.inner.write();
    let gm = g.models.entry(req.model.clone()).or_insert_with(|| GlobalModel { status: ModelStatus::Uninitialized, params: None, clients: vec![] });
    let client_id = Uuid::new_v4();
    let shard = gm.clients.len();
    let total_shards = shard + 1; // grow shards as clients join
    let info = ClientInfo { id: client_id, url: req.client_url, shard, total_shards };
    gm.clients.push(info);
    Json(RegisterResponse { client_id, shard, total_shards })
}

#[derive(Deserialize)]
struct GetQuery { model: String }

async fn get_model(State(state): State<ServerState>, axum::extract::Query(q): axum::extract::Query<GetQuery>) -> Json<GetResponse> {
    let g = state.inner.read();
    if let Some(gm) = g.models.get(&q.model) {
        return Json(GetResponse { status: gm.status.clone(), params: gm.params.clone() });
    }
    Json(GetResponse { status: ModelStatus::Uninitialized, params: None })
}

async fn init_model(State(state): State<ServerState>, Json(req): Json<InitRequest>) -> Json<GetResponse> {
    let dev = candle_core::Device::cuda_if_available(0).unwrap_or_else(|_| candle_core::Device::Cpu);
    let params = random_linear_params(&dev).expect("init params");
    let mut g = state.inner.write();
    let gm = g.models.entry(req.model.clone()).or_insert_with(|| GlobalModel { status: ModelStatus::Uninitialized, params: None, clients: vec![] });
    gm.params = Some(params.clone());
    gm.status = ModelStatus::Ready;
    Json(GetResponse { status: gm.status.clone(), params: gm.params.clone() })
}

async fn test_model(State(state): State<ServerState>, Json(req): Json<TestRequest>) -> Json<TestResponse> {
    let dev = candle_core::Device::cuda_if_available(0).unwrap_or_else(|_| candle_core::Device::Cpu);
    let g = state.inner.read();
    let Some(gm) = g.models.get(&req.model) else { return Json(TestResponse { accuracy: 0.0 }) };
    let Some(params) = gm.params.clone() else { return Json(TestResponse { accuracy: 0.0 }) };
    drop(g);

    // Build linear model and evaluate on full MNIST test set
    let lin = linear_from_params(&params, &dev).expect("linear");
    let m = candle_datasets::vision::mnist::load().expect("mnist");
    let test_images = m.test_images.to_device(&dev).expect("dev");
    let test_labels = m.test_labels.to_dtype(candle_core::DType::U32).expect("u32").to_device(&dev).expect("dev");
    let logits = candle_nn::Module::forward(&lin, &test_images).expect("fwd");
    let sum_ok = logits.argmax(candle_core::D::Minus1).expect("argmax")
        .eq(&test_labels).expect("eq")
        .to_dtype(candle_core::DType::F32).expect("f32")
        .sum_all().expect("sum")
        .to_scalar::<f32>().expect("scalar");
    let acc = sum_ok / test_labels.dims1().expect("dims") as f32;
    Json(TestResponse { accuracy: acc })
}

async fn train(State(state): State<ServerState>, Json(req): Json<TrainRequest>) -> Json<GetResponse> {
    {
        let mut g = state.inner.write();
        let gm = g.models.entry(req.model.clone()).or_insert_with(|| GlobalModel { status: ModelStatus::Uninitialized, params: None, clients: vec![] });
        gm.status = ModelStatus::Training;
    }
    let model_name = req.model.clone();
    let model_name_for_task = model_name.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        let state_for_update = state_clone.clone();
        if let Err(e) = run_rounds(state_clone, req).await {
            error!("training failed: {e:#}");
            let mut g = state_for_update.inner.write();
            if let Some(gm) = g.models.get_mut(&model_name_for_task) {
                gm.status = ModelStatus::Ready;
            }
        }
    });

    // Return current status
    let g = state.inner.read();
    let gm = g.models.get(&model_name);
    if let Some(gm) = gm {
        Json(GetResponse { status: gm.status.clone(), params: gm.params.clone() })
    } else {
        Json(GetResponse { status: ModelStatus::Uninitialized, params: None })
    }
}

async fn run_rounds(state: ServerState, req: TrainRequest) -> anyhow::Result<()> {
    use futures::future::join_all;
    let client = reqwest::Client::new();
    for round in 0..req.rounds {
        info!("Round {}", round + 1);
        // snapshot current params and clients
        let (params, clients) = {
            let g = state.inner.read();
            let gm = g.models.get(&req.model).expect("model exists");
            let params = gm.params.clone().expect("initialized");
            let clients = gm.clients.clone();
            (params, clients)
        };
        if clients.is_empty() { anyhow::bail!("no clients registered"); }
        let k = req.clients_per_round.min(clients.len());
        let selected = clients.into_iter().take(k).collect::<Vec<_>>();

        // Dispatch in parallel with unified shard assignment per round
        let futs = selected.into_iter().enumerate().map(|(idx, c)| {
            let body = TrainLocalRequest {
                model: req.model.clone(),
                params: params.clone(),
                epochs: req.local_epochs,
                learning_rate: req.learning_rate,
                shard: idx,
                total_shards: k,
            };
            let url = format!("{}/train_local", c.url.trim_end_matches('/'));
            client.post(url).json(&body).send()
        });
        let responses = join_all(futs).await;
        let mut collected: Vec<TrainLocalResponse> = Vec::new();
        for r in responses {
            match r {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let v: TrainLocalResponse = resp.json().await?;
                        collected.push(v);
                    }
                }
                Err(e) => { error!("client error: {e}"); }
            }
        }
        if collected.is_empty() { anyhow::bail!("no successful client updates"); }
        // Average params
        let mut w_acc = vec![0f32; LABELS * IMAGE_DIM];
        let mut b_acc = vec![0f32; LABELS];
        for upd in &collected {
            for (i, v) in upd.params.weight.iter().enumerate() { w_acc[i] += *v; }
            for (i, v) in upd.params.bias.iter().enumerate() { b_acc[i] += *v; }
        }
        let n = collected.len() as f32;
        for v in &mut w_acc { *v /= n; }
        for v in &mut b_acc { *v /= n; }
        let new_params = ModelParams { weight: w_acc, bias: b_acc, out_dim: LABELS, in_dim: IMAGE_DIM };
        // Update global model
        let mut g = state.inner.write();
        if let Some(gm) = g.models.get_mut(&req.model) {
            gm.params = Some(new_params);
        }
    }
    // Set READY
    let mut g = state.inner.write();
    if let Some(gm) = g.models.get_mut(&req.model) { gm.status = ModelStatus::Ready; }
    Ok(())
}