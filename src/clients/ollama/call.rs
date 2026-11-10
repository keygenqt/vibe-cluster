// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Request bodies and the non-stream call paths of the Ollama API.

use std::time::Duration;

use super::OllamaClient;
use crate::constants;
use crate::data::Prompt;
use crate::data::PromptStats;
use crate::error::ClusterError;
use crate::error::token_limit_message;
use crate::support::http_model_error;
use crate::utils::strip_data_uri;

impl OllamaClient {
    /// Executes a prompt: a conversation goes to /api/chat, a single
    /// prompt stays on /api/generate.
    pub(super) async fn call_impl(
        &self,
        prompt: &Prompt,
    ) -> Result<(String, String, PromptStats), ClusterError> {
        if prompt.history.is_empty() {
            self.call_generate(prompt).await
        } else {
            self.call_chat(prompt).await
        }
    }

    /// Sends a raw single-turn prompt to Ollama /api/generate and
    /// returns normalized response text.
    async fn call_generate(
        &self,
        prompt: &Prompt,
    ) -> Result<(String, String, PromptStats), ClusterError> {
        let url = format!("{}/api/generate", self.host);
        let request = self.generate_request(prompt, false);

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
            return Err(http_model_error(status, &body_text));
        }

        let mut full_response = String::new();
        let mut thinking_response = String::new();
        let mut hit_token_limit = false;
        let mut saw_json_chunk = false;
        let mut has_response_node = false;
        let mut prompt_eval_count = None;
        let mut prompt_eval_cached = None;
        let mut eval_count = None;

        // Deserializer chunk loop iterator processing
        let stream =
            serde_json::Deserializer::from_str(&body_text).into_iter::<serde_json::Value>();
        for value in stream {
            if let Ok(json_chunk) = value {
                saw_json_chunk = true;
                has_response_node |= json_chunk.get("response").is_some();

                if let Some(reason) = json_chunk["done_reason"].as_str() {
                    if reason == constants::TOKEN_LIMIT_REASON {
                        hit_token_limit = true;
                    }
                }
                if let Some(thinking_text) = json_chunk["thinking"].as_str() {
                    thinking_response.push_str(thinking_text);
                }
                if let Some(chunk_text) = json_chunk["response"].as_str() {
                    full_response.push_str(chunk_text);
                }
                prompt_eval_count = json_chunk["prompt_eval_count"]
                    .as_u64()
                    .or(prompt_eval_count);
                prompt_eval_cached = json_chunk["prompt_eval_cached_count"]
                    .as_u64()
                    .or(prompt_eval_cached);
                eval_count = json_chunk["eval_count"].as_u64().or(eval_count);
            }
        }

        if hit_token_limit && full_response.is_empty() {
            return Err(ClusterError::TokenLimitExceeded(token_limit_message(
                &self.model,
                self.params.num_predict,
            )));
        }

        if !has_response_node {
            if saw_json_chunk || body_text.trim().is_empty() {
                return Err(ClusterError::Parse(
                    "Ollama response misses the response node".into(),
                ));
            }

            // Fallback block for non-standard plain text response envelopes
            full_response = body_text;
        }

        // A plain-text fallback response carries no counters.
        let stats = ollama_stats(prompt_eval_count, prompt_eval_cached, eval_count);

        Ok((thinking_response, full_response, stats))
    }

    /// Sends the conversation to Ollama /api/chat — the native
    /// messages endpoint; /api/generate cannot carry history.
    async fn call_chat(
        &self,
        prompt: &Prompt,
    ) -> Result<(String, String, PromptStats), ClusterError> {
        let url = format!("{}/api/chat", self.host);
        let request = self.chat_request(prompt, false);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await?;

        let status = response.status();
        let body_text = response.text().await?;

        if !status.is_success() {
            return Err(http_model_error(status, &body_text));
        }

        let mut full_response = String::new();
        let mut thinking_response = String::new();
        let mut hit_token_limit = false;
        let mut has_message_node = false;
        let mut prompt_eval_count = None;
        let mut prompt_eval_cached = None;
        let mut eval_count = None;

        // the stream chunks mirror /api/generate, content sits one
        // level deeper — under "message"
        let stream =
            serde_json::Deserializer::from_str(&body_text).into_iter::<serde_json::Value>();
        for value in stream {
            if let Ok(json_chunk) = value {
                has_message_node |= json_chunk.get("message").is_some();

                if let Some(reason) = json_chunk["done_reason"].as_str() {
                    if reason == constants::TOKEN_LIMIT_REASON {
                        hit_token_limit = true;
                    }
                }
                if let Some(thinking_text) = json_chunk["message"]["thinking"].as_str() {
                    thinking_response.push_str(thinking_text);
                }
                if let Some(chunk_text) = json_chunk["message"]["content"].as_str() {
                    full_response.push_str(chunk_text);
                }
                prompt_eval_count = json_chunk["prompt_eval_count"]
                    .as_u64()
                    .or(prompt_eval_count);
                prompt_eval_cached = json_chunk["prompt_eval_cached_count"]
                    .as_u64()
                    .or(prompt_eval_cached);
                eval_count = json_chunk["eval_count"].as_u64().or(eval_count);
            }
        }

        if hit_token_limit && full_response.is_empty() {
            return Err(ClusterError::TokenLimitExceeded(token_limit_message(
                &self.model,
                self.params.num_predict,
            )));
        }

        if !has_message_node {
            return Err(ClusterError::Parse(
                "Ollama chat response misses the message node".into(),
            ));
        }

        let stats = ollama_stats(prompt_eval_count, prompt_eval_cached, eval_count);

        Ok((thinking_response, full_response, stats))
    }

    /// Builds the /api/generate request body; `stream` selects the mode.
    pub(super) fn generate_request(&self, prompt: &Prompt, stream: bool) -> serde_json::Value {
        let mut request = serde_json::json!({
            "model": self.model,
            "prompt": prompt.user,
            "stream": stream,
            "options": self.params_options(),
        });

        if let Some(system) = &prompt.system {
            request["system"] = serde_json::Value::String(system.to_string());
        }

        if let Some(images) = &prompt.images {
            // Ollama wants raw base64 without a data: URI prefix
            let raw: Vec<&str> = images.iter().map(|img| strip_data_uri(img)).collect();
            request["images"] = serde_json::to_value(raw).unwrap_or_default();
        }

        request
    }

    /// Builds the /api/chat request body — the native messages
    /// endpoint; /api/generate cannot carry history.
    pub(super) fn chat_request(&self, prompt: &Prompt, stream: bool) -> serde_json::Value {
        let mut messages = Vec::new();

        if let Some(system) = &prompt.system {
            messages.push(serde_json::json!({ "role": "system", "content": system }));
        }

        for message in &prompt.history {
            let mut message_json = serde_json::json!({
                "role": message.role,
                "content": message.content,
            });
            if !message.images.is_empty() {
                // Ollama wants raw base64 without a data: URI prefix
                let raw: Vec<&str> = message
                    .images
                    .iter()
                    .map(|img| strip_data_uri(img))
                    .collect();
                message_json["images"] = serde_json::to_value(raw).unwrap_or_default();
            }
            messages.push(message_json);
        }

        let mut last = serde_json::json!({ "role": "user", "content": prompt.user });
        if let Some(images) = &prompt.images {
            // Ollama wants raw base64 without a data: URI prefix
            let raw: Vec<&str> = images.iter().map(|img| strip_data_uri(img)).collect();
            last["images"] = serde_json::to_value(raw).unwrap_or_default();
        }
        messages.push(last);

        serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": self.params_options(),
        })
    }

    /// Builds the request `options` block from the generation
    /// parameters: a `None` stays out, the server default applies.
    fn params_options(&self) -> serde_json::Value {
        let mut options = serde_json::Map::new();
        if let Some(value) = self.params.temperature {
            options.insert("temperature".into(), value.into());
        }
        if let Some(value) = self.params.seed {
            options.insert("seed".into(), value.into());
        }
        if let Some(value) = self.params.top_p {
            options.insert("top_p".into(), value.into());
        }
        if let Some(value) = self.params.top_k {
            options.insert("top_k".into(), value.into());
        }
        if let Some(value) = self.params.min_p {
            options.insert("min_p".into(), value.into());
        }
        if let Some(value) = self.params.num_ctx {
            options.insert("num_ctx".into(), value.into());
        }
        if let Some(value) = self.params.num_predict {
            options.insert("num_predict".into(), value.into());
        }
        if let Some(stop) = self.params.stop.as_ref() {
            if !stop.is_empty() {
                options.insert("stop".into(), stop.clone().into());
            }
        }
        serde_json::Value::Object(options)
    }
}

/// Assembles the Ollama token counters into the report; `None` fields
/// mean the provider did not report them (older servers, plain-text
/// fallback). There is no thinking split: `eval_count` includes it.
pub(super) fn ollama_stats(
    prompt_eval_count: Option<u64>,
    prompt_eval_cached: Option<u64>,
    eval_count: Option<u64>,
) -> PromptStats {
    PromptStats::new(
        prompt_eval_cached.map(|value| value as usize),
        prompt_eval_count
            .map(|prompt| prompt.saturating_sub(prompt_eval_cached.unwrap_or(0)) as usize),
        eval_count.map(|value| value as usize),
        None,
    )
}
