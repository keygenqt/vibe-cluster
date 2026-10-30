//! Connection parameters for LLM providers.
//! Used by Cluster to create internal clients for each model.

use crate::provider::Provider;

/// Connection settings for a single model.
#[derive(Debug, Clone)]
pub struct ConnectionParams {
    /// Provider type: Ollama, DeepSeek, Qwen, etc.
    pub provider: Provider,
    /// API endpoint URL (e.g. http://localhost:11434 for Ollama, https://api.deepseek.com/v1 for DeepSeek).
    pub host: String,
    /// Model name (e.g. "qwen2.5:7b", "deepseek-chat").
    pub model: String,
    /// Temperature for generation (0.0-2.0). Lower = more deterministic.
    pub temperature: Option<f64>,
    /// Seed for reproducible outputs.
    pub seed: Option<u64>,
    /// Context window size in tokens.
    pub num_ctx: Option<usize>,
    /// Maximum tokens to generate.
    pub num_predict: Option<usize>,
    /// Request timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// API key for cloud providers (not needed for Ollama).
    pub api_key: Option<String>,
    /// Project/folder id for providers that route by it (Yandex).
    pub project: Option<String>,
    /// Number of parallel connections to this provider (default 1).
    pub parallel: usize,
}

/// Default connection: Ollama on localhost with 120s timeout, no model set.
impl Default for ConnectionParams {
    fn default() -> Self {
        Self {
            provider: Provider::Ollama,
            host: String::new(),
            model: String::new(),
            temperature: None,
            seed: None,
            num_ctx: None,
            num_predict: None,
            timeout_secs: Some(120),
            api_key: None,
            project: None,
            parallel: 1,
        }
    }
}
