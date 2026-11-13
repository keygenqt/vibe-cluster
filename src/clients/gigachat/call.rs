// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! OAuth token fetch and the call/stream paths of the GigaChat client.

use std::time::Duration;

use tokio::sync::mpsc;

use super::GigaChatClient;
use crate::constants;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptStats;
use crate::error::ClusterError;
use crate::error::error_chain;
use crate::error::token_limit_message;
use crate::support::build_messages;
use crate::support::cached_token;
use crate::support::http_model_error;
use crate::support::invalidate_token;
use crate::support::parse_usage;
use crate::support::spawn_sse_forwarder;
use crate::support::store_token;

/// OAuth endpoint issuing short-lived access tokens.
const OAUTH_URL: &str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
/// Default OpenAI-compatible API host (personal scope).
pub(crate) const DEFAULT_API_HOST: &str = "https://api.giga.chat/v1";
/// Cache the 30-minute token for 25 minutes — refresh 5 minutes
/// before the real expiry.
const TOKEN_LIFETIME: Duration = Duration::from_secs(25 * 60);

impl GigaChatClient {
    /// Sends a prompt to the OpenAI-compatible /chat/completions endpoint.
    pub(super) async fn call_impl(
        &self,
        prompt: &Prompt,
    ) -> Result<(String, String, PromptStats), ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        // Inline base64 in image_url is unverified for GigaChat —
        // the documented path is a /files upload with a file_id; test
        // live, on failure add the upload pre-step (call and call_stream).
        let mut request = serde_json::json!({
            "model": self.model,
            "messages": build_messages(prompt),
            "stream": false,
        });
        self.apply_params(&mut request);

        // One automatic retry with a fresh token when the cached one
        // expires between the refresh margin and the real 30-minute TTL.
        for attempt in 0..2 {
            let result = self.call_attempt(&url, &request).await;
            if attempt == 0 {
                if let Err(ClusterError::HttpStatus {
                    status: constants::STATUS_UNAUTHORIZED,
                    ..
                }) = &result
                {
                    invalidate_token(&self.auth_key).await;
                    continue;
                }
            }
            return result;
        }
        unreachable!("retry loop returns on both attempts")
    }

    /// One send attempt of `call_impl`: token, POST, parse.
    async fn call_attempt(
        &self,
        url: &str,
        request: &serde_json::Value,
    ) -> Result<(String, String, PromptStats), ClusterError> {
        let token = self.access_token().await?;

        let response = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("{} {}", constants::AUTH_BEARER, token),
            )
            .json(request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

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

        // Reasoning rides `reasoning_content` (v1 format) next to the answer.
        let thinking = message_node["reasoning_content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let raw = message_node["content"].as_str().unwrap_or("").to_string();

        let finish_reason = json["choices"]
            .get(0)
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(|reason| reason.as_str())
            .unwrap_or("");

        if finish_reason == constants::TOKEN_LIMIT_REASON && raw.is_empty() {
            return Err(ClusterError::TokenLimitExceeded(token_limit_message(
                &self.model,
                self.params.num_predict,
            )));
        }

        let stats = parse_usage(&json["usage"]);

        Ok((thinking, raw, stats))
    }

    /// Streams the response via SSE (stream=true). The OAuth token
    /// refresh retry mirrors `call_impl`. HTTP errors before the stream
    /// keep the Err return; a transport error mid-stream is sent through
    /// the channel before it closes.
    pub(super) async fn stream_impl(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        let mut request = serde_json::json!({
            "model": self.model,
            "messages": build_messages(prompt),
            "stream": true,
        });
        self.apply_params(&mut request);

        // No total timeout: a long generation must not be cut by one;
        // the cancel token (Cluster) stops runaway streams. The token
        // refresh retry mirrors `call_impl`.
        for attempt in 0..2 {
            let result = self.stream_attempt(&url, &request).await;
            if attempt == 0 {
                if let Err(ClusterError::HttpStatus {
                    status: constants::STATUS_UNAUTHORIZED,
                    ..
                }) = &result
                {
                    invalidate_token(&self.auth_key).await;
                    continue;
                }
            }
            return result;
        }
        unreachable!("retry loop returns on both attempts")
    }

    /// One send attempt of `call_stream_impl`: token, POST, forwarder.
    async fn stream_attempt(
        &self,
        url: &str,
        request: &serde_json::Value,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let token = self.access_token().await?;

        let response = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("{} {}", constants::AUTH_BEARER, token),
            )
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await?;
            return Err(http_model_error(status, &body_text));
        }

        Ok(spawn_sse_forwarder(
            response,
            token_limit_message(&self.model, self.params.num_predict),
        ))
    }

    /// Fills the params fields of the request body: a `None` stays
    /// out, the model default applies. The GigaChat schema has no
    /// stop/top_k/min_p; `num_predict` goes out as `max_tokens`,
    /// reasoning depth and its token cap as `reasoning_effort` /
    /// `reasoning_max_tokens` (v1 format).
    fn apply_params(&self, request: &mut serde_json::Value) {
        if let Some(value) = self.params.temperature {
            request["temperature"] = value.into();
        }
        if let Some(value) = self.params.seed {
            request["seed"] = value.into();
        }
        if let Some(value) = self.params.top_p {
            request["top_p"] = value.into();
        }
        if let Some(value) = self.params.num_predict {
            request["max_tokens"] = value.into();
        }
        if let Some(effort) = self.params.reasoning_effort {
            request["reasoning_effort"] = effort.as_str().into();
        }
        if let Some(value) = self.params.reasoning_max_tokens {
            request["reasoning_max_tokens"] = value.into();
        }
    }

    /// Returns a cached token or requests a fresh one via OAuth.
    pub(super) async fn access_token(&self) -> Result<String, ClusterError> {
        if let Some(token) = cached_token(&self.auth_key, TOKEN_LIFETIME).await {
            return Ok(token);
        }
        let token = self.request_token().await?;
        store_token(&self.auth_key, &token).await;
        Ok(token)
    }

    /// Requests a fresh access token via OAuth.
    async fn request_token(&self) -> Result<String, ClusterError> {
        let response = self
            .client
            .post(OAUTH_URL)
            .header("Authorization", format!("Basic {}", self.auth_key))
            .header("RqUID", uuid::Uuid::new_v4().to_string())
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .body("scope=GIGACHAT_API_PERS")
            .timeout(constants::OAUTH_TIMEOUT)
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
        json["access_token"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| ClusterError::Parse("GigaChat OAuth: no access_token".into()))
    }
}
