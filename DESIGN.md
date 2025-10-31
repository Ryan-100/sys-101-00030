# Federated Averaging (FedAvg) system design

This solution adds a networked parameter server and federated clients to the provided Candle MNIST template.

Goals
- Correct FedAvg: parameter server averages client-updated model parameters each round.
- Concurrency & safety: async HTTP with Axum, shared state behind RwLock to avoid races; training rounds run in a background task.
- Simplicity: HTTP/JSON (no gRPC), CPU-only Candle to avoid CUDA toolchain.

Components
- Parameter Server (binary subcommand `server`):
  - APIs
    - POST /register { client_url, model } -> { client_id, shard, total_shards }
    - POST /init { model } -> initializes a random global model
    - POST /train { model, rounds, clients_per_round, local_epochs, learning_rate } -> starts background FedAvg rounds
    - GET /get?model=NAME -> { status, params? }
    - POST /test { model } -> { accuracy }
  - State: map model_name -> { status, params, clients[] } protected by RwLock.
  - Training: in each round selects up to K registered clients, posts /train_local in parallel, averages returned weight and bias.

- Federated Client (binary subcommand `client`):
  - Optional Join: on startup can POST /register to a server, advertising its base URL.
  - APIs
    - POST /train_local with payload { params, epochs, lr, shard, total_shards }: trains locally on its MNIST shard, returns updated params and metrics.
    - GET /get -> local status and last params.
    - POST /test -> accuracy on full local test set.
  - Local state: status + last params behind RwLock.

Model & serialization
- Model supported: MNIST linear classifier W ∈ R[10×784], b ∈ R[10].
- Parameters serialized as JSON: flattened `weight: Vec<f32>`, `bias: Vec<f32>`, plus dims.
- Conversion helpers in `src/model.rs` create Candle Linear from params and extract params from a trained model.

Data partitioning
- IID shards using contiguous split of MNIST across the clients selected in a round. For each training round, the server assigns shard indices 0..K-1 to the K selected clients and sets `total_shards = K`, ensuring non-overlapping, consistent splits that reflect who actually participates that round (no stale totals from earlier registrations). Both train and test sets are partitioned consistently per shard.

Concurrency & synchronization
- Axum + Tokio. Server’s global state behind `parking_lot::RwLock` for low contention reads (GET) and safe writes (init/train updates).
- Training loop runs in `tokio::spawn`, executing client RPCs concurrently and averaging results after all responses.

Correctness expectations
- Local models differ due to different shards. Averaging improves global performance across rounds (with sufficient epochs/rounds and reasonable learning rates).
- Deterministic per-round shard assignment ensures consistent local datasets among the selected clients in that round.

Build/run
- Requirements: Rust stable. Candle runs on CPU (no CUDA required).
- Commands:
  - Start server: `cargo run --release -- server --bind 127.0.0.1:3000`
  - Start 2+ clients (different ports) and auto-register to server:
    - `cargo run --release -- client --bind 127.0.0.1:4001 --server http://127.0.0.1:3000`
    - `cargo run --release -- client --bind 127.0.0.1:4002 --server http://127.0.0.1:3000`
  - Initialize global model:
    - `curl -X POST http://127.0.0.1:3000/init -H "content-type: application/json" -d '{"model":"linear"}'`
  - Start federated training rounds (e.g., 3 rounds, 2 clients/round, 1 local epoch, lr=1.0):
    - `curl -X POST http://127.0.0.1:3000/train -H "content-type: application/json" -d '{"model":"linear","rounds":3,"clients_per_round":2,"local_epochs":1,"learning_rate":1.0}'`
  - Poll global status/params:
    - `curl http://127.0.0.1:3000/get?model=linear`
  - Test global model:
    - `curl -X POST http://127.0.0.1:3000/test -H "content-type: application/json" -d '{"model":"linear"}'`

Notes
- Original sequential demo preserved as `cargo run -- demo`.
- For real deployments consider weighted averaging by local sample counts and robustness to stragglers/timeouts.
