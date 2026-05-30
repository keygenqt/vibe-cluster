//! OpenAI-compatible API client implementation.
//! Supports DeepSeek, Qwen, Kimi and any other provider with OpenAI-compatible API.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::clients::client::Client;
use crate::error::ClusterError;
use crate::params::ConnectionParams;
use crate::provider::Provider;

/// Client for interacting with OpenAI-compatible APIs (DeepSeek, Qwen, etc.).
#[derive(Clone)]
pub struct OpenAiClient {
    client: HttpClient,
    host: String,
    model: String,
    api_key: String,
    temperature: f32,
    seed: u64,
    num_predict: usize,
    timeout_secs: u64,
    provider: Provider,
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
        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) => Ok(r.status().is_success()),
            Err(e) => Err(ClusterError::Http(format!(
                "{:?} unavailable: {}",
                self.provider(),
                e
            ))),
        }
    }

    /// Cloud models are always active (no loading/unloading).
    async fn is_active(&self) -> Result<bool, ClusterError> {
        Ok(true)
    }

    /// Cloud providers handle concurrency, always considered not busy.
    async fn is_busy(&self) -> Result<bool, ClusterError> {
        Ok(false)
    }

    /// Cloud models cannot be ejected (always available).
    async fn eject(&self) -> Result<(), ClusterError> {
        Ok(())
    }

    /// Sends a prompt to the OpenAI-compatible /chat/completions endpoint and returns raw response text.
    async fn call(&self, user: &str, system: Option<&str>) -> Result<String, reqwest::Error> {
        let url = format!("{}/chat/completions", self.host);
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": user
        }));

        let request = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.num_predict,
            "stream": false,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let json: serde_json::Value = response.json().await?;
        let raw = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        if raw.is_empty() {
            let alt = json["response"].as_str().unwrap_or("");
            return Ok(alt.to_string());
        }

        Ok(raw.to_string())
    }
}

impl OpenAiClient {
    /// Creates a new OpenAI-compatible client from connection parameters.
    pub fn new(params: ConnectionParams) -> Self {
        Self {
            client: HttpClient::new(),
            host: params.host,
            model: params.model,
            api_key: params.api_key.unwrap_or_default(),
            temperature: params.temperature.unwrap_or(0.7),
            seed: params.seed.unwrap_or(42),
            num_predict: params.num_predict.unwrap_or(2048),
            timeout_secs: params.timeout_secs.unwrap_or(120),
            provider: params.provider,
        }
    }
}
