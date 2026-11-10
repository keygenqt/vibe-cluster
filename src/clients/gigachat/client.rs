// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! The `GigaChatClient` entity and its `ClientTrait` impl: the trait
//! contract lives here, OAuth and call/stream paths in `requests`.

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
use crate::data::GigaChatParams;
use crate::data::ModelInfo;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptStats;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::support::http_model_error;

/// Client for the GigaChat API (Sber).
#[derive(Clone)]
pub struct GigaChatClient {
    /// Both GigaChat hosts serve the Russian GOST-signed chain,
    /// unverifiable by macOS TLS stacks — the client skips the
    /// server check (the `curl -k` behavior); traffic stays encrypted.
    pub(super) client: HttpClient,
    pub(super) rules: Arc<CapabilityRules>,
    pub(super) host: String,
    pub(super) model: String,
    /// Authorization key (Basic credentials) used to obtain access tokens.
    pub(super) auth_key: String,
    pub(super) params: GigaChatParams,
    pub(super) timeout_secs: u64,
    pub(super) bpe: Option<Bpe>,
    /// Capabilities, resolved once at construction (name-based rules).
    pub(super) capabilities: Capabilities,
}

#[async_trait]
impl ClientTrait for GigaChatClient {
    /// Returns the API host URL.
    fn host(&self) -> &str {
        &self.host
    }

    /// Returns Provider::GigaChat.
    fn provider(&self) -> Provider {
        Provider::GigaChat
    }

    /// The optional BPE estimator of this connection.
    fn bpe(&self) -> Option<Bpe> {
        self.bpe
    }

    /// Returns the model name (e.g. "GigaChat", "GigaChat-Pro").
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Checks if the API is reachable via the /models endpoint.
    async fn is_available(&self) -> Result<bool, ClusterError> {
        let token = self.access_token().await?;
        let url = format!("{}/models", self.host);

        let response = self
            .client
            .get(&url)
            .header(
                "Authorization",
                format!("{} {}", constants::AUTH_BEARER, token),
            )
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        Ok(response.status().is_success())
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
    /// Resolved once at construction: name-based rules — GigaChat-2-Lite
    /// and the first-gen base models are text-only.
    async fn capabilities(&self) -> Result<Capabilities, ClusterError> {
        Ok(self.capabilities.clone())
    }

    /// Lists models on the host. GigaChat /models reports only
    /// id/owned_by/created; capabilities come from the name rules,
    /// embedding from the catalog `type` field.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ClusterError> {
        let token = self.access_token().await?;
        let url = format!("{}/models", self.host);

        let response = self
            .client
            .get(&url)
            .header(
                "Authorization",
                format!("{} {}", constants::AUTH_BEARER, token),
            )
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        if !status.is_success() {
            return Err(http_model_error(status, &body_text));
        }

        let json: serde_json::Value = serde_json::from_str(&body_text)?;
        let Some(models) = json["data"].as_array() else {
            return Ok(Vec::new());
        };

        Ok(models
            .iter()
            .filter_map(|m| {
                let name = m["id"].as_str()?.to_string();
                let mut capabilities = self.rules.capabilities(&Provider::GigaChat, &name);
                // the catalog reports the kind; embedders are filtered out by the caller
                capabilities.abilities.embedding = Some(m["type"].as_str() == Some("embedder"));
                Some(ModelInfo {
                    owned_by: m["owned_by"].as_str().map(String::from),
                    created: m["created"].as_i64(),
                    capabilities,
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

    /// Streams the response via SSE (stream=true). The OAuth token
    /// refresh retry mirrors `call`. HTTP errors before the stream
    /// keep the Err return; a transport error mid-stream is sent through
    /// the channel before it closes.
    async fn call_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        self.stream_impl(prompt).await
    }
}

impl GigaChatClient {
    /// Creates a new GigaChat client from connection parameters and
    /// the generation parameters extracted by the clients factory.
    /// `connection.api_key` holds the Authorization key (Basic credentials).
    pub fn new(
        connection: Connection,
        params: GigaChatParams,
        rules: Arc<CapabilityRules>,
    ) -> Self {
        let host = if connection.host.is_empty() {
            DEFAULT_API_HOST.to_string()
        } else {
            connection.host
        };
        // Capabilities resolve once: name-based rules are the only
        // cloud capability source.
        let capabilities = rules.capabilities(&Provider::GigaChat, &connection.model);
        Self {
            client: Self::build_client(),
            rules,
            host,
            model: connection.model,
            auth_key: connection.api_key.unwrap_or_default(),
            params,
            timeout_secs: connection
                .timeout_secs
                .unwrap_or(constants::DEFAULT_CALL_TIMEOUT_SECS),
            bpe: connection.bpe,
            capabilities,
        }
    }

    /// Builds the client: both GigaChat hosts serve the Russian
    /// GOST-signed chain, unverifiable by macOS TLS stacks —
    /// rustls parses the chain, the verification is skipped
    /// (the `curl -k` behavior).
    fn build_client() -> HttpClient {
        HttpClient::builder()
            .use_rustls_tls()
            .danger_accept_invalid_certs(true)
            .build()
            .expect("static client config is valid")
    }
}
