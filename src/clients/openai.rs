//! OpenAI-compatible API client implementation.
//! Supports DeepSeek, Qwen, Moonshot and any other provider with OpenAI-compatible API.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::Prompt;
use crate::clients::client::Client;
use crate::clients::vision;
use crate::error::ClusterError;
use crate::models::ModelInfo;
use crate::params::ConnectionParams;
use crate::provider::Provider;

/// Client for interacting with OpenAI-compatible APIs (DeepSeek, Qwen, etc.).
#[derive(Clone)]
pub struct OpenAiClient {
    client: HttpClient,
    host: String,
    model: String,
    api_key: String,
    /// Project/folder id for providers that route by it (Yandex).
    project: Option<String>,
    temperature: f64,
    seed: u64,
    num_predict: usize,
    timeout_secs: u64,
    provider: Provider,
    /// Vision capability, detected once at construction (name-based).
    vision: bool,
}

#[async_trait]
impl Client for OpenAiClient {
    /// Returns the API host URL (e.g. "https://api.deepseek.com/v1").
    fn host(&self) -> &str {
        &self.host
    }

    /// Returns the provider set at creation (DeepSeek, Qwen, etc.).
    fn provider(&self) -> Provider {
        self.provider.clone()
    }

    /// Returns the model name (e.g. "deepseek-chat", "qwen-max").
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Checks if the cloud API is reachable via /models endpoint.
    async fn is_available(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/models", self.host);

        // Network errors are automatically transformed into ClusterError::Http via ? operator
        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_value())
            .timeout(Duration::from_secs(5))
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

    /// Checks if the model supports vision (image input).
    /// Cached at construction: none of the supported cloud providers expose
    /// modality info via their OpenAI-compatible API, so detection is
    /// name-based (see clients::vision).
    async fn is_vision(&self) -> Result<bool, ClusterError> {
        Ok(self.vision)
    }

    /// Lists models on the host. OpenAI-compatible /models reports only
    /// id/owned_by/created — capabilities stay `None`.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ClusterError> {
        let url = format!("{}/models", self.host);
        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_value())
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        if !status.is_success() {
            if let Ok(err_json) = serde_json::from_str::<serde_json::Value>(&body_text) {
                if let Some(msg) = err_json["error"]["message"]
                    .as_str()
                    .or_else(|| err_json["error"].as_str())
                {
                    return Err(ClusterError::Model(format!("HTTP {} - {}", status, msg)));
                }
            }
            return Err(ClusterError::Model(format!(
                "HTTP {} - {}",
                status, body_text
            )));
        }

        // Yandex does not expose a model catalog via API — static list
        // mirroring the Model Gallery (only -lite is curl-verified).
        if self.provider == Provider::Yandex {
            return Ok(["yandexgpt-5-lite", "yandexgpt-5-pro"]
                .iter()
                .map(|name| ModelInfo {
                    name: name.to_string(),
                    digest: None,
                    remote_model: None,
                    remote_host: None,
                    owned_by: Some("yandex".into()),
                    modified_at: None,
                    created: None,
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
                    embedding: None,
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
                    embedding: None,
                })
            })
            .collect())
    }

    /// Cloud models cannot be ejected (always available).
    async fn eject(&self) -> Result<(), ClusterError> {
        Ok(())
    }

    /// Sends a prompt to the OpenAI-compatible /chat/completions endpoint and returns raw response text.
    async fn call(&self, prompt: &Prompt) -> Result<(String, String), ClusterError> {
        let url = format!("{}/chat/completions", self.host);
        let mut messages = Vec::new();

        if let Some(system) = &prompt.system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }

        // prior turns go as native messages: each role gets the chat
        // template treatment, the prefix stays cache-stable
        for message in &prompt.history {
            messages.push(serde_json::json!({
                "role": message.role,
                "content": message.content,
            }));
        }

        let user_content = if let Some(images) = &prompt.images {
            let mut content: Vec<serde_json::Value> =
                vec![serde_json::json!({"type": "text", "text": prompt.user})];
            for img in images {
                // Prompt.images accepts raw base64 or a ready-made data: URI;
                // each client normalizes the payload for its provider.
                let url = if img.starts_with("data:") {
                    img.clone()
                } else {
                    format!("data:{};base64,{}", vision::mime_from_base64(img), img)
                };
                content.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": url}
                }));
            }
            serde_json::Value::Array(content)
        } else {
            serde_json::Value::String(prompt.user.clone())
        };

        messages.push(serde_json::json!({ "role": "user", "content": user_content }));

        let request = serde_json::json!({
            "model": self.model_uri(),
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.num_predict,
            "stream": false,
        });

        let mut builder = self
            .client
            .post(&url)
            .header("Authorization", self.auth_value())
            .header("Content-Type", "application/json");
        // Yandex routes the request to the project's folder.
        if let (Provider::Yandex, Some(project)) = (&self.provider, &self.project) {
            builder = builder.header("OpenAI-Project", project);
        }
        let response = builder
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        // Hard boundary verification: catch OpenAI/vLLM context limits and API errors
        if !status.is_success() {
            if let Ok(err_json) = serde_json::from_str::<serde_json::Value>(&body_text) {
                // Support both standard OpenAI nested error maps and flat error objects
                if let Some(msg) = err_json["error"]["message"]
                    .as_str()
                    .or_else(|| err_json["error"].as_str())
                {
                    return Err(ClusterError::Model(format!("HTTP {} - {}", status, msg)));
                }
            }
            return Err(ClusterError::Model(format!(
                "HTTP {} - {}",
                status, body_text
            )));
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

        if finish_reason == "length" && raw.is_empty() {
            return Err(ClusterError::TokenLimitExceeded(format!(
                "Model '{}' hit output token limit (num_predict: {}).",
                self.model, self.num_predict
            )));
        }

        // Fallback for vLLM/Ollama OpenAI-compatible response structures
        if raw.is_empty() {
            let alt = json["response"].as_str().unwrap_or("");
            if !alt.is_empty() {
                raw = alt.to_string();
            }
        }

        Ok((thinking.to_string(), raw))
    }
}

impl OpenAiClient {
    /// Authorization header value: Yandex wants "Api-Key", the rest — "Bearer".
    fn auth_value(&self) -> String {
        match self.provider {
            Provider::Yandex => format!("Api-Key {}", self.api_key),
            _ => format!("Bearer {}", self.api_key),
        }
    }

    /// Yandex models are addressed as URIs: gpt://<folder>/<name>.
    fn model_uri(&self) -> String {
        match (&self.provider, &self.project) {
            (Provider::Yandex, Some(project)) if !self.model.starts_with("gpt://") => {
                format!("gpt://{}/{}", project, self.model)
            }
            _ => self.model.clone(),
        }
    }

    /// Creates a new OpenAI-compatible client from connection parameters.
    pub fn new(params: ConnectionParams) -> Self {
        let temp = params.temperature.unwrap_or(0.7) as f64;
        // Detect vision before moving fields out of params.
        // Name-based; update clients::vision if a provider adds a model
        // that does not match the existing patterns.
        let vision = vision::model_supports_vision(&params.provider, &params.model);
        Self {
            client: HttpClient::new(),
            host: params.host,
            model: params.model,
            api_key: params.api_key.unwrap_or_default(),
            project: params.project.clone(),
            temperature: (temp * 100.0).round() / 100.0,
            seed: params.seed.unwrap_or(42),
            num_predict: params.num_predict.unwrap_or(2048),
            timeout_secs: params.timeout_secs.unwrap_or(120),
            provider: params.provider,
            vision,
        }
    }
}
