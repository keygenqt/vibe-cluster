// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! The `OllamaClient` entity and its `ClientTrait` impl: the trait contract
//! lives here, the call/stream/capability kitchen lives in the sibling
//! submodules.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use tokio::sync::mpsc;

use crate::capability::CapabilityRules;
use crate::client::ClientTrait;
use crate::constants;
use crate::data::Bpe;
use crate::data::Capabilities;
use crate::data::Connection;
use crate::data::ModelInfo;
use crate::data::OllamaParams;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptStats;
use crate::data::Provider;
use crate::error::ClusterError;

/// Client for interacting with Ollama API.
#[derive(Clone)]
pub struct OllamaClient {
    pub(super) client: HttpClient,
    pub(super) rules: Arc<CapabilityRules>,
    pub(super) host: String,
    pub(super) model: String,
    pub(super) params: OllamaParams,
    pub(super) timeout_secs: u64,
    pub(super) bpe: Option<Bpe>,
}

#[async_trait]
impl ClientTrait for OllamaClient {
    /// Returns the Ollama host URL (e.g. "http://localhost:11434").
    fn host(&self) -> &str {
        &self.host
    }

    /// Returns Provider::Ollama.
    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    /// The optional BPE estimator of this connection.
    fn bpe(&self) -> Option<Bpe> {
        self.bpe
    }

    /// Returns the model name (e.g. "qwen2.5-coder:3b-instruct").
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Checks if Ollama server is reachable via /api/tags endpoint.
    async fn is_available(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/api/tags", self.host);

        // Network errors are automatically transformed into ClusterError::Http via ? operator
        let response = self
            .client
            .get(&url)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        // Return true if the server responded with a successful status code
        Ok(response.status().is_success())
    }

    /// Checks if the model is loaded in Ollama memory via /api/ps.
    /// NOTE: /api/show cannot be used here — it succeeds for models that
    /// are merely present on disk, not loaded.
    async fn is_active(&self) -> Result<bool, ClusterError> {
        if !self.is_available().await? {
            return Ok(false);
        }

        let url = format!("{}/api/ps", self.host);
        let response = self
            .client
            .get(&url)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ClusterError::Model(format!(
                "Ollama /api/ps returned status: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response.json().await?;

        // Ollama reports full names in /api/ps (e.g. "llava:latest"),
        // the config may use the short form — compare with an implicit
        // ":latest" on both sides.
        if let Some(models) = json["models"].as_array() {
            fn strip_latest(s: &str) -> &str {
                s.strip_suffix(":latest").unwrap_or(s)
            }
            return Ok(models.iter().any(|m| {
                m["name"]
                    .as_str()
                    .is_some_and(|name| strip_latest(name) == strip_latest(&self.model))
            }));
        }

        Ok(false)
    }

    /// Ollama does not expose a "busy" state: it serializes requests per
    /// loaded model internally. Always reports not busy; the router should
    /// rely on Ollama's own queueing.
    async fn is_busy(&self) -> Result<bool, ClusterError> {
        Ok(false)
    }

    /// Returns the model capabilities: live from /api/show (a local
    /// model can be re-pulled under the same name at any time), empty
    /// fields fall back to the capability rules.
    async fn capabilities(&self) -> Result<Capabilities, ClusterError> {
        let show = self.show_model(&self.model).await?;
        Ok(self.show_capabilities(&show, &self.model))
    }

    /// Lists all models on the host. Static facts come from /api/tags;
    /// capabilities and context length from /api/show per model — on
    /// failure or older servers they stay `None`.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ClusterError> {
        let url = format!("{}/api/tags", self.host);
        let response = self
            .client
            .get(&url)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ClusterError::Model(format!(
                "Ollama /api/tags returned status: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response.json().await?;
        let Some(models) = json["models"].as_array() else {
            return Ok(Vec::new());
        };

        let mut result = Vec::new();
        for model in models {
            let Some(name) = model["name"].as_str() else {
                continue;
            };
            let details = &model["details"];

            let mut info = ModelInfo {
                digest: model["digest"].as_str().map(String::from),
                remote_model: model["remote_model"].as_str().map(String::from),
                remote_host: model["remote_host"].as_str().map(String::from),
                modified_at: model["modified_at"].as_str().map(String::from),
                format: details["format"].as_str().map(String::from),
                family: details["family"].as_str().map(String::from),
                families: details["families"].as_array().map(|families| {
                    families
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
                params: details["parameter_size"].as_str().map(String::from),
                quantization: details["quantization_level"].as_str().map(String::from),
                size: model["size"].as_u64(),
                ..ModelInfo::new(name)
            };

            // A failed /api/show keeps capabilities default — the provider did not say.
            if let Ok(show) = self.show_model(name).await {
                info.capabilities = self.show_capabilities(&show, name);
                info.context_length = show["model_info"].as_object().and_then(|map| {
                    map.iter()
                        .find(|(key, _)| key.ends_with(".context_length"))
                        .and_then(|(_, value)| value.as_u64())
                });
            }

            result.push(info);
        }
        Ok(result)
    }

    /// Unloads the model from Ollama memory via /api/generate with keep_alive=0.
    async fn eject(&self) -> Result<(), ClusterError> {
        let url = format!("{}/api/generate", self.host);
        let request = serde_json::json!({
            "model": self.model,
            "prompt": "",
            "keep_alive": 0,
        });

        // Operator ? automatically transforms transport errors into ClusterError::Http
        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        // Check status code strictly using the same logic as other clients
        if !response.status().is_success() {
            return Err(ClusterError::Model(format!(
                "Ollama eject returned status: {}",
                response.status()
            )));
        }

        // Explicitly consume bytes to free up the HTTP connection pool safely
        let _ = response.bytes().await?;

        Ok(())
    }

    /// Sends a raw prompt to Ollama: a conversation goes to
    /// /api/chat, a single prompt stays on /api/generate. Returns the
    /// raw text with the provider token report.
    async fn call(&self, prompt: &Prompt) -> Result<(String, String, PromptStats), ClusterError> {
        self.call_impl(prompt).await
    }

    /// Streams the response: Ollama sends NDJSON chunks with
    /// stream=true. HTTP errors before the stream keep the Err
    /// return; a transport error mid-stream is sent through the channel
    /// before it closes.
    async fn call_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        self.stream_impl(prompt).await
    }
}

impl OllamaClient {
    /// Creates a new Ollama client from connection parameters and the
    /// generation parameters extracted by the clients factory.
    pub fn new(connection: Connection, params: OllamaParams, rules: Arc<CapabilityRules>) -> Self {
        Self {
            client: HttpClient::new(),
            rules,
            host: connection.host,
            model: connection.model,
            params,
            timeout_secs: connection
                .timeout_secs
                .unwrap_or(constants::DEFAULT_CALL_TIMEOUT_SECS),
            bpe: connection.bpe,
        }
    }
}
