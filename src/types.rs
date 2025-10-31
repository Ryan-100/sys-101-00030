use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::model::ModelParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub client_url: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub client_id: Uuid,
    pub shard: usize,
    pub total_shards: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitRequest {
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRequest {
    pub model: String,
    pub rounds: usize,
    pub clients_per_round: usize,
    pub local_epochs: usize,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainLocalRequest {
    pub model: String,
    pub params: ModelParams,
    pub epochs: usize,
    pub learning_rate: f64,
    pub shard: usize,
    pub total_shards: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainLocalResponse {
    pub params: ModelParams,
    pub train_loss: f32,
    pub test_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus { Uninitialized, Training, Ready }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetResponse {
    pub status: ModelStatus,
    pub params: Option<ModelParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRequest { pub model: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResponse { pub accuracy: f32 }