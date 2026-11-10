// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Batch execution options for controlling model selection,
//! concurrency, retries, and provider filtering.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::client::ClientTrait;
use crate::constants;
use crate::data::CancelToken;
use crate::data::Provider;

/// Options for batch execution. Cloning the options clones the
/// cancellation token — clones stay bound to the same cancel.
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// Which models to use: all, local only, cloud only, specific model names.
    pub models: ModelFilter,
    /// Force specific provider.
    pub provider: Option<Provider>,
    /// Maximum number of attempts per prompt; zero still allows one attempt.
    /// `None` — a single attempt, no retries.
    pub retries: Option<usize>,
    /// Ensure full GPU memory for the model: eject and reload between
    /// prompts for clean context. Only meaningful for Ollama — cloud
    /// providers are always available and ignore the flag. Honest
    /// only with `parallel = 1` on the connection — sibling
    /// connections can re-load the model in the settle window after
    /// the eject.
    pub exclusive: bool,
    /// Cancellation handle: cancel() stops pending attempts and the
    /// remaining prompts of the batch. Default — a fresh token, never cancelled.
    pub cancel: CancelToken,
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
            retries: Some(constants::DEFAULT_RETRIES),
            exclusive: false,
            cancel: CancelToken::default(),
        }
    }
}

impl BatchOptions {
    /// Checks if a client should be included based on model filter and provider constraints.
    pub async fn matches_client(&self, client: &Arc<Mutex<Box<dyn ClientTrait>>>) -> bool {
        let client = client.lock().await;

        if let Some(ref provider) = self.provider {
            if client.provider() != *provider {
                return false;
            }
        }

        match &self.models {
            ModelFilter::All => true,
            ModelFilter::LocalOnly => matches!(client.provider(), Provider::Ollama),
            ModelFilter::CloudOnly => !matches!(client.provider(), Provider::Ollama),
            ModelFilter::ByName(names) => names.iter().any(|name| name == client.model_name()),
        }
    }
}
