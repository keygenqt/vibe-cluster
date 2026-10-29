//! vibe-cluster — LLM cluster library.
//! Manages multiple LLM providers and distributes prompts across them.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use vibe_cluster::{Cluster, ConnectionParams, Provider};
//!
//! let connections = vec![ConnectionParams {
//!     provider: Provider::Ollama,
//!     host: "http://localhost:11434".into(),
//!     model: "qwen2.5-coder:3b-instruct".into(),
//!     ..Default::default()
//! }];
//!
//! let cluster = Cluster::new(connections).unwrap();
//! let result = cluster.call("Explain async/await").await.unwrap();
//! println!("{}", result.text.unwrap());
//! ```

mod clients;
mod cluster;
mod error;
mod models;
mod options;
mod params;
mod prompt;
mod provider;
mod result;
mod utils;

pub use clients::client::Client;
pub use cluster::Cluster;
pub use error::ClusterError;
pub use models::ModelInfo;
pub use options::BatchOptions;
pub use options::ModelFilter;
pub use params::ConnectionParams;
pub use prompt::HistoryMessage;
pub use prompt::Prompt;
pub use provider::Provider;
pub use result::PromptResult;

pub use clients::create_client;
pub use utils::normalize_text;
