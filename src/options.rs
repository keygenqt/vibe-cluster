//! Batch execution options for controlling model selection,
//! concurrency, retries, and provider filtering.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{clients::client::Client, provider::Provider};

/// Options for batch execution.
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// Which models to use: all, local only, cloud only, specific model names.
    pub models: ModelFilter,
    /// Force specific provider.
    pub provider: Option<Provider>,
    /// Number of retries per prompt.
    pub retries: Option<usize>,
    /// Ensure full GPU memory for the model
    pub exclusive: bool,
}

#[derive(Debug, Clone)]
pub enum ModelFilter {
    /// Use all available models.
    All,
    /// Use only local models (Ollama).
    LocalOnly,
    /// Use only cloud models (DeepSeek, Qwen, etc.).
    CloudOnly,
    /// Use specific model names.
    ByName(Vec<String>),
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            models: ModelFilter::All,
            provider: None,
            retries: Some(5),
            exclusive: false,
        }
    }
}

impl BatchOptions {
    /// Checks if a client should be included based on model filter and provider constraints.
    pub fn matches_client(&self, client: &Arc<Mutex<Box<dyn Client>>>) -> bool {
        let client = client.try_lock();
        let Ok(client) = client else { return false };

        if let Some(ref provider) = self.provider {
            if client.provider() != *provider {
                return false;
            }
        }

        match &self.models {
            ModelFilter::All => true,
            ModelFilter::LocalOnly => matches!(client.provider(), Provider::Ollama),
            ModelFilter::CloudOnly => !matches!(client.provider(), Provider::Ollama),
            ModelFilter::ByName(names) => names.contains(&client.model_name().to_string()),
        }
    }
}
