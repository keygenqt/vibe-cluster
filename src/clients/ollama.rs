//! Ollama API client implementation.
//! Uses Ollama's native /api/generate endpoint.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use uuid::Uuid;

use crate::clients::client::Client;
use crate::error::ClusterError;
use crate::params::ConnectionParams;
use crate::provider::Provider;

/// Client for interacting with Ollama API.
#[derive(Clone)]
pub struct OllamaClient {
    client: HttpClient,
    host: String,
    model: String,
    temperature: f32,
    seed: u64,
    num_ctx: usize,
    num_predict: usize,
    timeout_secs: u64,
}

#[async_trait]
impl Client for OllamaClient {
    /// Returns the Ollama host URL (e.g. "http://localhost:11434").
    fn host(&self) -> &str {
        &self.host
    }

    /// Returns Provider::Ollama.
    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    /// Returns the model name (e.g. "qwen2.5-coder:3b-instruct").
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Checks if Ollama server is reachable via /api/tags endpoint.
    async fn is_available(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/api/tags", self.host);
        match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) => Ok(r.status().is_success()),
            Err(e) => Err(ClusterError::Http(format!("Ollama unavailable: {}", e))),
        }
    }

    /// Checks if the model is loaded in Ollama memory via /api/show.
    async fn is_active(&self) -> Result<bool, ClusterError> {
        self.is_available().await?;

        let url = format!("{}/api/show", self.host);
        let request = serde_json::json!({
            "name": self.model,
        });
        match self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) => Ok(r.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Checks if the model is currently processing a request via /api/ps.
    async fn is_busy(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/api/ps", self.host);
        match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) => {
                let json = r.json::<serde_json::Value>().await.map_err(|e| {
                    ClusterError::Parse(format!("Ollama /api/ps parse error: {}", e))
                })?;
                if let Some(models) = json["models"].as_array() {
                    return Ok(models
                        .iter()
                        .any(|m| m["name"].as_str() == Some(&self.model)));
                }
                Ok(false)
            }
            Err(e) => Err(ClusterError::Http(format!(
                "Ollama is_busy check failed: {}",
                e
            ))),
        }
    }

    /// Unloads the model from Ollama memory via /api/generate with keep_alive=0.
    async fn eject(&self) -> Result<(), ClusterError> {
        let url = format!("{}/api/generate", self.host);
        let request = serde_json::json!({
            "model": self.model,
            "prompt": "",
            "keep_alive": 0,
        });
        match self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(r) => Err(ClusterError::Model(format!(
                "Ollama eject returned status: {}",
                r.status()
            ))),
            Err(e) => Err(ClusterError::Http(format!("Ollama eject failed: {}", e))),
        }
    }

    /// Sends a raw prompt to Ollama /api/generate and returns normalized response text.
    async fn call(&self, prompt: &str) -> Result<String, reqwest::Error> {
        let url = format!("{}/api/generate", self.host);
        let request = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "session_id": Uuid::new_v4().to_string(),
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            }
        });

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let json: serde_json::Value = response.json().await?;
        let raw = json["response"].as_str().unwrap_or("");
        Ok(raw.to_string())
    }
}

impl OllamaClient {
    /// Creates a new Ollama client from connection parameters.
    pub fn new(params: ConnectionParams) -> Self {
        Self {
            client: HttpClient::new(),
            host: params.host,
            model: params.model,
            temperature: params.temperature.unwrap_or(0.7),
            seed: params.seed.unwrap_or(42),
            num_ctx: params.num_ctx.unwrap_or(4096),
            num_predict: params.num_predict.unwrap_or(2048),
            timeout_secs: params.timeout_secs.unwrap_or(120),
        }
    }
}
