// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Common trait for all internal HTTP clients.

use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::constants;
use crate::data::Bpe;
use crate::data::CancelToken;
use crate::data::Capabilities;
use crate::data::ModelInfo;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptResult;
use crate::data::PromptStats;
use crate::data::Provider;
use crate::error::ClusterError;

/// Client trait — one implementation per provider API.
#[async_trait]
pub trait ClientTrait: Send + Sync {
    /// Host URL of this client (e.g. "http://localhost:11434").
    fn host(&self) -> &str;

    /// Provider type for this client.
    fn provider(&self) -> Provider;

    /// The optional BPE estimator of this connection.
    fn bpe(&self) -> Option<Bpe>;

    /// Model name for this client.
    fn model_name(&self) -> &str;

    /// Check if the provider is reachable and model is available.
    async fn is_available(&self) -> Result<bool, ClusterError>;

    /// Check if the model is currently loaded in memory.
    async fn is_active(&self) -> Result<bool, ClusterError>;

    /// Check if the model is currently processing another request.
    async fn is_busy(&self) -> Result<bool, ClusterError>;

    /// Returns the model capabilities. Detection is live for Ollama
    /// (`/api/show`) and rule-based for the cloud providers; a failed
    /// check propagates as `Err`.
    async fn capabilities(&self) -> Result<Capabilities, ClusterError>;

    /// Lists models available on this provider's host (host-level: the
    /// configured model is ignored). Facts the provider does not
    /// report stay `None`.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ClusterError>;

    /// Unload the model from memory (free VRAM).
    async fn eject(&self) -> Result<(), ClusterError>;

    /// Low-level call — sends raw prompt, returns raw text with the
    /// provider-reported token report (`duration_ms` stays `0`, the
    /// executor measures the whole call).
    async fn call(&self, prompt: &Prompt) -> Result<(String, String, PromptStats), ClusterError>;

    /// Streams the response as thinking/text deltas over the channel;
    /// the channel closes when the response is complete. Default
    /// implementation wraps `call` — one burst, no live deltas.
    async fn call_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let (sender, receiver) = mpsc::channel(constants::STREAM_CHANNEL_SIZE);

        let (thinking, text, stats) = self.call(prompt).await?;
        if !thinking.is_empty() {
            let _ = sender.send(Ok(PromptEvent::Thinking(thinking))).await;
        }
        let _ = sender.send(Ok(PromptEvent::Text(text))).await;
        let _ = sender
            .send(Ok(PromptEvent::Usage {
                cached_tokens: stats.cached_tokens,
                input_tokens: stats.input_tokens,
                output_tokens: stats.output_tokens,
                thinking_tokens: stats.thinking_tokens,
            }))
            .await;

        Ok(receiver)
    }

    /// Executes a prompt with retries, normalization, and error
    /// classification. A cancelled token cuts the request — before any
    /// attempt, mid-call (the dropped HTTP future closes the
    /// connection), and between retries (checked at the loop top after
    /// the backoff sleep).
    async fn execute(
        &self,
        prompt: &Prompt,
        retries: usize,
        cancel: &CancelToken,
    ) -> Result<PromptResult, ClusterError> {
        let start = Instant::now();
        let mut attempts = 0;

        loop {
            if cancel.is_cancelled() {
                return Err(ClusterError::Cancelled);
            }

            let raw_cancel = cancel.raw();
            let raw = tokio::select! {
                _ = raw_cancel.cancelled() => return Err(ClusterError::Cancelled),
                raw = self.call(prompt) => raw,
            };
            attempts += 1;

            match raw {
                Ok((thinking_text, response_text, mut stats)) => {
                    stats.duration_ms = start.elapsed().as_millis() as u64;
                    if let Some(bpe) = self.bpe() {
                        stats.estimate_missing(bpe, prompt, &thinking_text, &response_text);
                    }
                    return Ok(PromptResult {
                        prompt: prompt.clone(),
                        provider: self.provider(),
                        host: self.host().to_string(),
                        model: self.model_name().to_string(),
                        stats: Some(stats),
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
                    // Retry only transient transport failures and server responses.
                    let retryable = match &e {
                        ClusterError::Http(reqwest_err) => {
                            reqwest_err.is_timeout() || reqwest_err.is_connect()
                        }
                        ClusterError::HttpStatus { status, .. } => {
                            constants::RETRYABLE_STATUS_RANGE.contains(status)
                        }
                        _ => false,
                    };

                    if retryable && attempts < retries {
                        log::warn!(
                            "{:?}/{}: attempt {}/{} failed, retrying: {}",
                            self.provider(),
                            self.model_name(),
                            attempts,
                            retries,
                            e
                        );
                        tokio::time::sleep(constants::RETRY_BACKOFF).await;
                        continue;
                    }

                    log::error!(
                        "{:?}/{}: failed after {} attempts: {}",
                        self.provider(),
                        self.model_name(),
                        attempts,
                        e
                    );
                    return Err(e);
                }
            }
        }
    }
}
