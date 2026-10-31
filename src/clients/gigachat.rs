// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! GigaChat API client implementation.
//! OAuth access token from Sber (30 min TTL, cached), then
//! OpenAI-compatible chat completions on api.giga.chat.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::client::Client;
use crate::data::ConnectionParams;
use crate::data::ModelInfo;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::error::error_chain;
use crate::support::spawn_sse_chunk_forwarder;
use tokio::sync::mpsc;

/// OAuth endpoint issuing short-lived access tokens.
const OAUTH_URL: &str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
/// Default OpenAI-compatible API host (personal scope).
const DEFAULT_API_HOST: &str = "https://api.giga.chat/v1";
/// Refresh the cached token 5 minutes before its 30-minute expiry.
const TOKEN_MARGIN_SECS: u64 = 25 * 60;

/// Process-wide token cache: one OAuth request per auth key per TTL,
/// shared across every GigaChatClient instance.
static TOKEN_CACHE: std::sync::LazyLock<tokio::sync::Mutex<Option<(String, String, Instant)>>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(None));

/// Client for the GigaChat API (Sber).
#[derive(Clone)]
pub struct GigaChatClient {
    /// Both GigaChat hosts serve the Russian GOST-signed chain,
    /// unverifiable by macOS TLS stacks — the client skips the
    /// server check (the `curl -k` behavior); traffic stays encrypted.
    client: HttpClient,
    host: String,
    model: String,
    /// Authorization key (Basic credentials) used to obtain access tokens.
    auth_key: String,
    temperature: f64,
    seed: u64,
    num_predict: usize,
    timeout_secs: u64,
}

#[async_trait]
impl Client for GigaChatClient {
    /// Returns the API host URL.
    fn host(&self) -> &str {
        &self.host
    }

    /// Returns Provider::GigaChat.
    fn provider(&self) -> Provider {
        Provider::GigaChat
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
            .header("Authorization", format!("Bearer {}", token))
            .timeout(Duration::from_secs(5))
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

    /// No vision-capable models are served on the personal scope.
    async fn is_vision(&self) -> Result<bool, ClusterError> {
        Ok(false)
    }

    /// Lists models on the host. GigaChat /models reports only
    /// id/owned_by/created — capabilities stay `None`.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ClusterError> {
        let token = self.access_token().await?;
        let url = format!("{}/models", self.host);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .timeout(Duration::from_secs(5))
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
                Some(ModelInfo {
                    name,
                    digest: None,
                    remote_model: None,
                    remote_host: None,
                    owned_by: m["owned_by"].as_str().map(String::from),
                    modified_at: None,
                    created: m["created"].as_i64(),
                    format: None,
                    family: None,
                    families: None,
                    params: None,
                    quantization: None,
                    size: None,
                    context_length: None,
                    vision: None,
                    tools: None,
                    thinking: None,
                    // the catalog reports the kind; embedders are
                    // filtered out by the studio layer
                    embedding: Some(m["type"].as_str() == Some("embedder")),
                })
            })
            .collect())
    }

    /// Cloud models cannot be ejected (always available).
    async fn eject(&self) -> Result<(), ClusterError> {
        Ok(())
    }

    /// Sends a prompt to the OpenAI-compatible /chat/completions endpoint.
    async fn call(&self, prompt: &Prompt) -> Result<(String, String), ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        let mut messages = Vec::new();

        if let Some(system) = &prompt.system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }

        for message in &prompt.history {
            messages.push(serde_json::json!({
                "role": message.role,
                "content": message.content,
            }));
        }

        messages.push(serde_json::json!({ "role": "user", "content": prompt.user }));

        let request = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.num_predict,
            "stream": false,
        });

        // One automatic retry with a fresh token when the cached one
        // expires between the refresh margin and the real 30-minute TTL.
        for attempt in 0..2 {
            let token = self.access_token().await?;

            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .header("Content-Type", "application/json")
                .json(&request)
                .timeout(Duration::from_secs(self.timeout_secs))
                .send()
                .await?;

            let status = response.status();
            let body_text = response.text().await?;

            if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                self.invalidate_token().await;
                continue;
            }

            if !status.is_success() {
                return Err(http_model_error(status, &body_text));
            }

            let json: serde_json::Value = serde_json::from_str(&body_text)?;

            let message_node = json["choices"]
                .get(0)
                .and_then(|choice| choice.get("message"))
                .ok_or_else(|| {
                    ClusterError::Parse("GigaChat response misses choices or message node".into())
                })?;

            let raw = message_node["content"].as_str().unwrap_or("").to_string();

            let finish_reason = json["choices"]
                .get(0)
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|reason| reason.as_str())
                .unwrap_or("");

            if finish_reason == "length" && raw.is_empty() {
                return Err(ClusterError::TokenLimitExceeded(format!(
                    "Model '{}' hit output token limit (num_predict: {}).",
                    self.model, self.num_predict
                )));
            }

            return Ok((String::new(), raw));
        }

        unreachable!("retry loop returns or continues on both attempts")
    }

    /// Streams the response via SSE (stream=true). The OAuth token
    /// refresh retry mirrors `call`. HTTP errors before the stream
    /// keep the Err return; a transport error mid-stream is sent through
    /// the channel before it closes.
    async fn call_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        let mut messages = Vec::new();

        if let Some(system) = &prompt.system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }

        for message in &prompt.history {
            messages.push(serde_json::json!({
                "role": message.role,
                "content": message.content,
            }));
        }

        messages.push(serde_json::json!({ "role": "user", "content": prompt.user }));

        let request = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.num_predict,
            "stream": true,
        });

        // One automatic retry with a fresh token when the cached one
        // expires between the refresh margin and the real 30-minute TTL.
        for attempt in 0..2 {
            let token = self.access_token().await?;

            // No total timeout: a long generation must not be cut by one;
            // the cancel token (Cluster) stops runaway streams.
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await?;

            let status = response.status();
            if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                self.invalidate_token().await;
                continue;
            }

            if !status.is_success() {
                let body_text = response.text().await?;
                return Err(http_model_error(status, &body_text));
            }

            return Ok(spawn_sse_chunk_forwarder(response));
        }

        unreachable!("retry loop returns or continues on both attempts")
    }
}

impl GigaChatClient {
    /// Creates a new GigaChat client from connection parameters.
    /// `params.api_key` holds the Authorization key (Basic credentials).
    pub fn new(params: ConnectionParams) -> Self {
        let temp = params.temperature.unwrap_or(0.7) as f64;
        let host = if params.host.is_empty() {
            DEFAULT_API_HOST.to_string()
        } else {
            params.host
        };
        Self {
            client: Self::build_client(),
            host,
            model: params.model,
            auth_key: params.api_key.unwrap_or_default(),
            temperature: (temp * 100.0).round() / 100.0,
            seed: params.seed.unwrap_or(42),
            num_predict: params.num_predict.unwrap_or(2048),
            timeout_secs: params.timeout_secs.unwrap_or(120),
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

    /// Returns a cached token or requests a fresh one via OAuth.
    /// The cache is process-wide and keyed by the auth key; the
    /// mutex is held across the request — single-flight.
    async fn access_token(&self) -> Result<String, ClusterError> {
        let mut guard = TOKEN_CACHE.lock().await;

        if let Some((key, token, obtained)) = &*guard {
            if *key == self.auth_key && obtained.elapsed() < Duration::from_secs(TOKEN_MARGIN_SECS)
            {
                return Ok(token.clone());
            }
        }

        let response = self
            .client
            .post(OAUTH_URL)
            .header("Authorization", format!("Basic {}", self.auth_key))
            .header("RqUID", uuid::Uuid::new_v4().to_string())
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .body("scope=GIGACHAT_API_PERS")
            .timeout(Duration::from_secs(10))
            .send()
            .await;

        let response = match response {
            Ok(response) => response,
            // reqwest hides the TLS/network cause behind a generic
            // top-level message — log the full source chain
            Err(error) => {
                log::error!("GigaChat OAuth request failed: {}", error_chain(&error));
                return Err(error.into());
            }
        };

        let status = response.status();
        let body_text = response.text().await?;

        if !status.is_success() {
            log::error!("GigaChat OAuth failed: HTTP {} - {}", status, body_text);
            return Err(ClusterError::Model(format!(
                "GigaChat OAuth {} - {}",
                status, body_text
            )));
        }

        let json: serde_json::Value = serde_json::from_str(&body_text)?;
        let token = json["access_token"]
            .as_str()
            .ok_or_else(|| ClusterError::Parse("GigaChat OAuth: no access_token".into()))?
            .to_string();

        *guard = Some((self.auth_key.clone(), token.clone(), Instant::now()));
        Ok(token)
    }

    /// Drops the cached token so the next call requests a fresh one.
    async fn invalidate_token(&self) {
        *TOKEN_CACHE.lock().await = None;
    }
}

/// Builds a model error from a failed HTTP response body, digging
/// the error message out of the OpenAI-style JSON envelopes.
fn http_model_error(status: reqwest::StatusCode, body_text: &str) -> ClusterError {
    if let Ok(err_json) = serde_json::from_str::<serde_json::Value>(body_text) {
        if let Some(msg) = err_json["error"]["message"]
            .as_str()
            .or_else(|| err_json["error"].as_str())
        {
            return ClusterError::HttpStatus {
                status: status.as_u16(),
                message: msg.to_string(),
            };
        }
    }
    ClusterError::HttpStatus {
        status: status.as_u16(),
        message: body_text.to_string(),
    }
}
