// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Request building of the OpenAI-compatible client: the auth
//! builder, generation params, model addressing.

use std::time::Duration;

use tokio::sync::mpsc;

use super::OpenAiClient;
use crate::constants;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptStats;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::error::token_limit_message;
use crate::support::build_messages;
use crate::support::http_model_error;
use crate::support::parse_usage;
use crate::support::spawn_sse_forwarder;

/// Default host for the OpenAI provider when none is configured.
pub(crate) const DEFAULT_API_HOST: &str = "https://api.openai.com/v1";

impl OpenAiClient {
    /// Executes a prompt against /chat/completions and returns the raw response text.
    pub(super) async fn call_impl(
        &self,
        prompt: &Prompt,
    ) -> Result<(String, String, PromptStats), ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        let mut request = serde_json::json!({
            "model": self.model_uri(),
            "messages": build_messages(prompt),
            "stream": false,
        });
        self.apply_params(&mut request);

        let response = self
            .request_builder(reqwest::Method::POST, &url)
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        // Hard boundary verification: catch OpenAI/vLLM context limits and API errors
        if !status.is_success() {
            return Err(http_model_error(status, &body_text));
        }

        let json: serde_json::Value = serde_json::from_str(&body_text)?;

        // Safe extraction of the first message node using options to prevent runtime panics
        let message_node = json["choices"]
            .get(0)
            .and_then(|choice| choice.get("message"))
            .ok_or_else(|| {
                ClusterError::Parse("OpenAI response misses choices or message node".into())
            })?;

        // Extract reasoning/thinking log if available (e.g. from DeepSeek-R1 or OpenAI o1/o3)
        let thinking = message_node["reasoning_content"]
            .as_str()
            .or_else(|| message_node["reasoning"].as_str())
            .unwrap_or("");

        // Extract core content text
        let mut raw = message_node["content"].as_str().unwrap_or("").to_string();

        // Safe finish reason boundary evaluation
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

        // Fallback for vLLM/Ollama OpenAI-compatible response structures
        if raw.is_empty() {
            let alt = json["response"].as_str().unwrap_or("");
            if !alt.is_empty() {
                raw = alt.to_string();
            }
        }

        let stats = parse_usage(&json["usage"]);

        Ok((thinking.to_string(), raw, stats))
    }

    /// Streams the response via SSE (stream=true). HTTP errors before the stream
    /// keep the Err return; a transport error mid-stream is sent through the
    /// channel before it closes.
    pub(super) async fn stream_impl(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        let mut request = serde_json::json!({
            "model": self.model_uri(),
            "messages": build_messages(prompt),
            "stream": true,
        });
        self.apply_params(&mut request);

        // api.openai.com reports the stream usage only with the option;
        // every other verified host sends the usage (or ignores the
        // unknown field) without it.
        if self.provider == Provider::OpenAI {
            request["stream_options"] = serde_json::json!({ "include_usage": true });
        }

        // No total timeout: a long generation must not be cut by one;
        // the cancel token (Cluster) stops runaway streams.
        let response = self
            .request_builder(reqwest::Method::POST, &url)
            .json(&request)
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

    /// A POST/GET builder with the auth header in place. The
    /// Content-Type goes out with `.json()` — an explicit header on
    /// the GET endpoints (/models) would be noise.
    pub(super) fn request_builder(
        &self,
        method: reqwest::Method,
        url: &str,
    ) -> reqwest::RequestBuilder {
        let mut builder = self
            .client
            .request(method, url)
            .header("Authorization", self.auth_value());
        // Yandex routes the request to the project's folder.
        if let (Provider::Yandex, Some(project)) = (&self.provider, &self.project) {
            builder = builder.header("OpenAI-Project", project);
        }
        builder
    }

    /// Authorization header value: Yandex wants "Api-Key", the rest — "Bearer".
    pub(super) fn auth_value(&self) -> String {
        match self.provider {
            Provider::Yandex => format!("Api-Key {}", self.api_key),
            _ => format!("{} {}", constants::AUTH_BEARER, self.api_key),
        }
    }

    /// Yandex models are addressed as URIs: gpt://<folder>/<name>.
    pub(super) fn model_uri(&self) -> String {
        match (&self.provider, &self.project) {
            (Provider::Yandex, Some(project)) if !self.model.starts_with("gpt://") => {
                format!("gpt://{}/{}", project, self.model)
            }
            _ => self.model.clone(),
        }
    }

    /// Fills the params fields of the request body: a `None` stays
    /// out, the provider default applies; `num_predict` goes out as
    /// `max_tokens`.
    pub(super) fn apply_params(&self, request: &mut serde_json::Value) {
        if let Some(value) = self.params.temperature {
            request["temperature"] = value.into();
        }
        if let Some(value) = self.params.top_p {
            request["top_p"] = value.into();
        }
        if let Some(value) = self.params.seed {
            request["seed"] = value.into();
        }
        if let Some(value) = self.params.top_k {
            request["top_k"] = value.into();
        }
        if let Some(value) = self.params.min_p {
            request["min_p"] = value.into();
        }
        if let Some(value) = self.params.num_predict {
            request["max_tokens"] = value.into();
        }
        if let Some(stop) = self.params.stop.as_ref() {
            if !stop.is_empty() {
                request["stop"] = stop.clone().into();
            }
        }
    }
}
