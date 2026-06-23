//! Common trait for all internal HTTP clients.

use std::time::{Duration, Instant};

use crate::provider::Provider;
use crate::result::PromptResult;
use crate::{ClusterError, Prompt};
use async_trait::async_trait;

/// Internal client trait — not exposed to library users.
#[async_trait]
pub trait Client: Send + Sync {
    /// Host URL of this client (e.g. "http://localhost:11434").
    fn host(&self) -> &str;

    /// Provider type for this client.
    fn provider(&self) -> Provider;

    /// Model name for this client.
    fn model_name(&self) -> &str;

    /// Check if the provider is reachable and model is available.
    async fn is_available(&self) -> Result<bool, ClusterError>;

    /// Check if the model is currently loaded in memory.
    async fn is_active(&self) -> Result<bool, ClusterError>;

    /// Check if the model is currently processing another request.
    async fn is_busy(&self) -> Result<bool, ClusterError>;

    /// Checks if the model supports vision (image input).
    async fn is_vision(&self) -> Result<bool, ClusterError>;

    /// Unload the model from memory (free VRAM).
    async fn eject(&self) -> Result<(), ClusterError>;

    /// Low-level call — sends raw prompt, returns raw text.
    async fn call(&self, prompt: &Prompt) -> Result<(String, String), ClusterError>;

    /// Executes a prompt with retries, normalization, and error classification.
    async fn execute(&self, prompt: &Prompt, retries: usize) -> Result<PromptResult, ClusterError> {
        let start = Instant::now();
        let mut attempts = 0;

        loop {
            let raw = self.call(prompt).await;
            attempts += 1;

            match raw {
                Ok((thinking_text, response_text)) => {
                    return Ok(PromptResult {
                        prompt: prompt.clone(),
                        provider: self.provider(),
                        host: self.host().to_string(),
                        model: self.model_name().to_string(),
                        duration_ms: start.elapsed().as_millis() as u64,
                        thinking: if thinking_text.is_empty() {
                            None
                        } else {
                            Some(thinking_text)
                        },
                        text: Some(response_text),
                        error: None,
                    });
                }
                Err(e) => {
                    // Handle standard transport and HTTP network errors if applicable
                    if let ClusterError::Http(ref reqwest_err) = e {
                        if reqwest_err.is_timeout() && attempts < retries {
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            continue;
                        }
                        if reqwest_err.is_connect() && attempts < retries {
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            continue;
                        }

                        if let Some(status) = reqwest_err.status() {
                            if status.is_server_error() && attempts < retries {
                                tokio::time::sleep(Duration::from_millis(500)).await;
                                continue;
                            }
                            return Ok(PromptResult {
                                prompt: prompt.clone(),
                                provider: self.provider(),
                                host: self.host().to_string(),
                                model: self.model_name().to_string(),
                                duration_ms: start.elapsed().as_millis() as u64,
                                thinking: None,
                                text: None,
                                error: Some(ClusterError::Model(format!(
                                    "HTTP {}: {}",
                                    status.as_u16(),
                                    reqwest_err
                                ))),
                            });
                        }
                    }

                    // Retry loop fallback for remaining attempts before total exhaustion
                    if attempts < retries {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        continue;
                    }

                    // Return the structured cluster error directly without temporary allocations
                    return Err(e);
                }
            }
        }
    }
}
