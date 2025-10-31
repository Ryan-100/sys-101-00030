use std::sync::Arc;

use axum::{routing::{get, post}, Json, Router, extract::State};
use parking_lot::RwLock;
use tracing::info;
use tokio::net::TcpListener;
use candle_nn::Module;

use crate::model::{local_train_linear, ModelParams};
use crate::types::*;

#[derive(Clone)]
struct ClientState { inner: Arc<RwLock<Inner>> }

struct Inner {
    status: ModelStatus,
    params: Option<ModelParams>,
    shard: usize,
    total_shards: usize,
}

impl ClientState {
    fn new() -> Self { Self { inner: Arc::new(RwLock::new(Inner { status: ModelStatus::Uninitialized, params: None, shard: 0, total_shards: 1 })) } }
}

pub async fn run(bind: String, server_url: Option<String>, model: String) -> anyhow::Result<()> {
    // Optionally register with server
    if let Some(srv) = &server_url {
        let client = reqwest::Client::new();
        let body = RegisterRequest { client_url: format!("http://{}", bind), model: model.clone() };
        let resp = client.post(format!("{}/register", srv.trim_end_matches('/'))).json(&body).send().await?;
        if resp.status().is_success() { info!("Registered with server {}", srv); }
    }
    let state = ClientState::new();
    let app = Router::new()
        .route("/train_local", post(train_local))
        .route("/get", get(get_local))
        .route("/test", post(test_local))
        .with_state(state);
    info!("Client listening on {}", bind);
    let listener = TcpListener::bind(&bind).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn train_local(State(state): State<ClientState>, Json(req): Json<TrainLocalRequest>) -> Json<TrainLocalResponse> {
    let dev = candle_core::Device::cuda_if_available(0).unwrap_or_else(|_| candle_core::Device::Cpu);
    let (new_params, stats) = local_train_linear(req.params, req.epochs, req.learning_rate, req.shard, req.total_shards, &dev).expect("local train");
    {
        let mut g = state.inner.write();
        g.status = ModelStatus::Ready;
        g.params = Some(new_params.clone());
        g.shard = req.shard;
        g.total_shards = req.total_shards;
    }
    Json(TrainLocalResponse { params: new_params, train_loss: stats.final_loss, test_accuracy: stats.test_accuracy })
}

async fn get_local(State(state): State<ClientState>) -> Json<GetResponse> {
    let g = state.inner.read();
    Json(GetResponse { status: g.status.clone(), params: g.params.clone() })
}

async fn test_local(State(state): State<ClientState>) -> Json<TestResponse> {
    let dev = candle_core::Device::cuda_if_available(0).unwrap_or_else(|_| candle_core::Device::Cpu);
    let g = state.inner.read();
    if let Some(params) = g.params.clone() {
        drop(g);
        let lin = crate::model::linear_from_params(&params, &dev).expect("linear");
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
    } else {
        Json(TestResponse { accuracy: 0.0 })
    }
}