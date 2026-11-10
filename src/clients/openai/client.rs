// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! The `OpenAiClient` entity and its `ClientTrait` impl: the trait contract
//! lives here, call paths and request building live in `call`.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use tokio::sync::mpsc;

use super::call::DEFAULT_API_HOST;
use crate::capability::CapabilityRules;
use crate::client::ClientTrait;
use crate::constants;
use crate::data::Bpe;
use crate::data::Capabilities;
use crate::data::Connection;
use crate::data::ModelInfo;
use crate::data::OpenAiParams;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptStats;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::support::http_model_error;

/// Client for interacting with OpenAI-compatible APIs (DeepSeek, Qwen, etc.).
#[derive(Clone)]
pub struct OpenAiClient {
    pub(super) client: HttpClient,
    pub(super) rules: Arc<CapabilityRules>,
    pub(super) host: String,
    pub(super) model: String,
    pub(super) api_key: String,
    /// Project/folder id for providers that route by it (Yandex).
    pub(super) project: Option<String>,
    pub(super) params: OpenAiParams,
    pub(super) timeout_secs: u64,
    pub(super) provider: Provider,
    /// Capabilities, resolved once at construction (name-based rules).
    pub(super) capabilities: Capabilities,
    pub(super) bpe: Option<Bpe>,
}

#[async_trait]
impl ClientTrait for OpenAiClient {
    /// Returns the API host URL (e.g. "https://api.deepseek.com/v1").
    fn host(&self) -> &str {
        &self.host
    }

    /// Returns the provider set at creation (OpenAI, DeepSeek, Qwen, etc.).
    fn provider(&self) -> Provider {
        self.provider
    }

    /// The optional BPE estimator of this connection.
    fn bpe(&self) -> Option<Bpe> {
        self.bpe
    }

    /// Returns the model name (e.g. "gpt-4o", "deepseek-chat", "qwen-max").
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Checks if the cloud API is reachable via /models endpoint.
    async fn is_available(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/models", self.host);

        // Network errors are automatically transformed into ClusterError::Http via ? operator
        let response = self
            .request_builder(reqwest::Method::GET, &url)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        // Yandex's OpenAI layer has no /models route: a 404 here still
        // proves the host is reachable and the key routing works.
        Ok(response.status().is_success()
            || (self.provider == Provider::Yandex
                && response.status() == reqwest::StatusCode::NOT_FOUND))
    }

    /// Cloud models are always active (no loading/unloading).
    async fn is_active(&self) -> Result<bool, ClusterError> {
        Ok(true)
    }

    /// Cloud providers handle concurrency, always considered not busy.
    async fn is_busy(&self) -> Result<bool, ClusterError> {
        Ok(false)
    }

    /// Returns the model capabilities.
    /// Resolved once at construction: none of the supported cloud providers
    /// expose modality info via their OpenAI-compatible API, so detection
    /// is name-based (capability rules).
    async fn capabilities(&self) -> Result<Capabilities, ClusterError> {
        Ok(self.capabilities.clone())
    }

    /// Lists models on the host. OpenAI-compatible /models reports only
    /// id/owned_by/created; capabilities come from the name rules.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ClusterError> {
        let url = format!("{}/models", self.host);
        let response = self
            .request_builder(reqwest::Method::GET, &url)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        if !status.is_success()
            && !(self.provider == Provider::Yandex && status == reqwest::StatusCode::NOT_FOUND)
        {
            return Err(http_model_error(status, &body_text));
        }

        // Yandex does not expose a model catalog via API — static list
        // mirroring the Model Gallery (only -lite is curl-verified).
        if self.provider == Provider::Yandex {
            return Ok(["yandexgpt-5-lite", "yandexgpt-5-pro"]
                .iter()
                .map(|name| ModelInfo {
                    owned_by: Some("yandex".into()),
                    capabilities: self.rules.capabilities(&self.provider, name),
                    ..ModelInfo::new(*name)
                })
                .collect());
        }

        let json: serde_json::Value = serde_json::from_str(&body_text)?;
        let Some(models) = json["data"].as_array() else {
            return Ok(Vec::new());
        };

        Ok(models
            .iter()
            .filter_map(|m| {
                let name = m["id"].as_str()?.to_string();
                Some(ModelInfo {
                    owned_by: m["owned_by"].as_str().map(String::from),
                    created: m["created"].as_i64(),
                    capabilities: self.rules.capabilities(&self.provider, &name),
                    ..ModelInfo::new(name)
                })
            })
            .collect())
    }

    /// Cloud models cannot be ejected (always available).
    async fn eject(&self) -> Result<(), ClusterError> {
        Ok(())
    }

    /// Sends a prompt to the OpenAI-compatible /chat/completions endpoint
    /// and returns raw response text with the provider token report.
    async fn call(&self, prompt: &Prompt) -> Result<(String, String, PromptStats), ClusterError> {
        self.call_impl(prompt).await
    }

    /// Streams the response via SSE (stream=true). HTTP errors before the stream
    /// keep the Err return; a transport error mid-stream is sent through the
    /// channel before it closes.
    async fn call_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        self.stream_impl(prompt).await
    }
}

impl OpenAiClient {
    /// Creates a new OpenAI-compatible client from connection
    /// parameters and the generation parameters extracted by the
    /// clients factory. An empty host falls back to the OpenAI API
    /// host for the OpenAI provider; the other providers keep the
    /// host as given.
    pub fn new(connection: Connection, params: OpenAiParams, rules: Arc<CapabilityRules>) -> Self {
        let host = if connection.provider == Provider::OpenAI && connection.host.is_empty() {
            DEFAULT_API_HOST.to_string()
        } else {
            connection.host
        };
        // Capabilities resolve once: name-based rules are the only
        // cloud capability source.
        let capabilities = rules.capabilities(&connection.provider, &connection.model);
        Self {
            client: HttpClient::new(),
            rules,
            host,
            model: connection.model,
            api_key: connection.api_key.unwrap_or_default(),
            project: connection.project.clone(),
            params,
            timeout_secs: connection
                .timeout_secs
                .unwrap_or(constants::DEFAULT_CALL_TIMEOUT_SECS),
            provider: connection.provider,
            capabilities,
            bpe: connection.bpe,
        }
    }
}
