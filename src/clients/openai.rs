//! OpenAI-compatible API client implementation.
//! Supports DeepSeek, Qwen, Kimi and any other provider with OpenAI-compatible API.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::Prompt;
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

        // Network errors are automatically transformed into ClusterError::Http via ? operator
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        // Return true if the cloud server responded with a successful status code
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

    /// Checks if the model supports vision (image input) via API, with name pattern fallback.
    async fn is_vision(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/models/{}", self.host, self.model);
        // Try API first
        if let Ok(r) = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            if let Ok(json) = r.json::<serde_json::Value>().await {
                if let Some(modalities) = json["modalities"].as_array() {
                    return Ok(modalities.iter().any(|v| v.as_str() == Some("image")));
                }
            }
        }
        // Fallback to name pattern
        let name = self.model.to_lowercase();
        Ok(name.contains("vl") || name.contains("vision") || name.contains("gpt-4o"))
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

        let user_content = if let Some(images) = &prompt.images {
            let mut content: Vec<serde_json::Value> =
                vec![serde_json::json!({"type": "text", "text": prompt.user})];
            for img in images {
                content.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:image/png;base64,{}", img)}
                }));
            }
            serde_json::Value::Array(content)
        } else {
            serde_json::Value::String(prompt.user.clone())
        };

        messages.push(serde_json::json!({ "role": "user", "content": user_content }));

        let request = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.num_predict,
            "stream": false,
        });

        // Transport and HTTP error codes are automatically transformed into ClusterError::Http via ?
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let response = response.error_for_status()?;
        let json: serde_json::Value = response.json().await?;

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
