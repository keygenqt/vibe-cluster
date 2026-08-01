//! Ollama API client implementation.
//! Uses Ollama's native /api/generate endpoint.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client as HttpClient;

use crate::Prompt;
use crate::clients::client::Client;
use crate::clients::vision;
use crate::error::ClusterError;
use crate::params::ConnectionParams;
use crate::provider::Provider;

/// Client for interacting with Ollama API.
#[derive(Clone)]
pub struct OllamaClient {
    client: HttpClient,
    host: String,
    model: String,
    temperature: f64,
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

        // Network errors are automatically transformed into ClusterError::Http via ? operator
        let response = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
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
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ClusterError::Model(format!(
                "Ollama /api/ps returned status: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response.json().await?;

        // Check if the current model is present in the loaded models list
        if let Some(models) = json["models"].as_array() {
            return Ok(models.iter().any(|m| {
                m["name"]
                    .as_str()
                    .map(|name| model_names_match(name, &self.model))
                    .unwrap_or(false)
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

    /// Checks if the model supports vision (image input) via /api/show.
    /// Live check (not cached): a local model can be re-pulled under the
    /// same name at any time.
    async fn is_vision(&self) -> Result<bool, ClusterError> {
        let url = format!("{}/api/show", self.host);
        let request = serde_json::json!({ "name": self.model });

        // Network errors are automatically transformed into ClusterError::Http via ? operator
        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        // Check for HTTP success status before parsing the response body
        if !response.status().is_success() {
            return Err(ClusterError::Model(format!(
                "Ollama /api/show returned status: {}",
                response.status()
            )));
        }

        // Parse JSON safely using implemented transformation traits
        let json: serde_json::Value = response.json().await?;

        // Modern Ollama versions expose an explicit capabilities array
        let capabilities = json["capabilities"].as_array();
        let is_vision_capable = capabilities
            .map(|c| c.iter().any(|v| v.as_str() == Some("vision")))
            .unwrap_or(false);

        if is_vision_capable {
            return Ok(true);
        }

        // Fallback for older Ollama versions: vision models carry a CLIP
        // encoder, a dedicated projector, or a known vision family.
        let families = json["details"]["families"].as_array();
        let has_vision_family = families
            .map(|f| {
                f.iter().any(|v| {
                    matches!(
                        v.as_str(),
                        Some("clip") | Some("mllama") | Some("qwen2vl") | Some("qwen3vl")
                    )
                })
            })
            .unwrap_or(false);

        // projector_info is a TOP-LEVEL field of /api/show, next to model_info
        let has_projector = json["projector_info"].is_object();

        Ok(has_vision_family || has_projector)
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
            .timeout(Duration::from_secs(5))
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

    /// Sends a raw prompt to Ollama /api/generate and returns normalized response text.
    async fn call(&self, prompt: &Prompt) -> Result<(String, String), ClusterError> {
        let url = format!("{}/api/generate", self.host);
        let mut request = serde_json::json!({
            "model": self.model,
            "prompt": prompt.user,
            "stream": false,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            }
        });

        if let Some(system) = &prompt.system {
            request["system"] = serde_json::Value::String(system.to_string());
        }

        if let Some(images) = &prompt.images {
            // Ollama wants raw base64 without a data: URI prefix
            let raw: Vec<&str> = images
                .iter()
                .map(|img| vision::strip_data_uri(img))
                .collect();
            request["images"] = serde_json::to_value(raw).unwrap_or_default();
        }

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        // Hard boundary verification: catch API errors and context overflows before stream fallback
        if !status.is_success() {
            // Safely extract deeply nested error messages from escaped JSON string containers
            if let Ok(top_json) = serde_json::from_str::<serde_json::Value>(&body_text) {
                if let Some(error_payload) = top_json["error"].as_str() {
                    if let Ok(inner_json) = serde_json::from_str::<serde_json::Value>(error_payload)
                    {
                        if let Some(msg) = inner_json["error"]["message"].as_str() {
                            return Err(ClusterError::Model(format!("HTTP {} - {}", status, msg)));
                        }
                    }
                } else if let Some(msg) = top_json["error"]["message"]
                    .as_str()
                    .or_else(|| top_json["error"].as_str())
                {
                    return Err(ClusterError::Model(format!("HTTP {} - {}", status, msg)));
                }
            }
            return Err(ClusterError::Model(format!(
                "HTTP {} - {}",
                status, body_text
            )));
        }

        let mut full_response = String::new();
        let mut thinking_response = String::new();
        let mut hit_token_limit = false;

        // Deserializer chunk loop iterator processing
        let stream =
            serde_json::Deserializer::from_str(&body_text).into_iter::<serde_json::Value>();
        for value in stream {
            if let Ok(json_chunk) = value {
                if let Some(reason) = json_chunk["done_reason"].as_str() {
                    if reason == "length" {
                        hit_token_limit = true;
                    }
                }
                if let Some(thinking_text) = json_chunk["thinking"].as_str() {
                    thinking_response.push_str(thinking_text);
                }
                if let Some(chunk_text) = json_chunk["response"].as_str() {
                    full_response.push_str(chunk_text);
                }
            }
        }

        if hit_token_limit && full_response.is_empty() {
            return Err(ClusterError::TokenLimitExceeded(format!(
                "Model '{}' hit output token limit (num_predict: {}).",
                self.model, self.num_predict
            )));
        }

        // Fallback block for non-standard plain text response envelopes
        if full_response.is_empty() && !body_text.contains("\"response\":\"\"") {
            full_response = body_text;
        }

        Ok((thinking_response, full_response))
    }
}

/// Ollama reports full model names in /api/ps (e.g. "llava:latest"),
/// while users may configure the short form ("llava").
/// Compare names with an implicit ":latest" suffix on both sides.
fn model_names_match(a: &str, b: &str) -> bool {
    fn strip_latest(s: &str) -> &str {
        s.strip_suffix(":latest").unwrap_or(s)
    }
    strip_latest(a) == strip_latest(b)
}

impl OllamaClient {
    /// Creates a new Ollama client from connection parameters.
    pub fn new(params: ConnectionParams) -> Self {
        let temp = params.temperature.unwrap_or(0.7) as f64;
        Self {
            client: HttpClient::new(),
            host: params.host,
            model: params.model,
            temperature: (temp * 100.0).round() / 100.0,
            seed: params.seed.unwrap_or(42),
            num_ctx: params.num_ctx.unwrap_or(4096),
            num_predict: params.num_predict.unwrap_or(2048),
            timeout_secs: params.timeout_secs.unwrap_or(120),
        }
    }
}
