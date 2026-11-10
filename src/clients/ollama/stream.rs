// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Streaming call paths of the Ollama API and the NDJSON chunk
//! forwarder into `PromptEvent`s.

use futures_util::StreamExt;
use reqwest::Response;
use tokio::sync::mpsc;

use super::OllamaClient;
use super::call::ollama_stats;
use crate::constants;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::error::ClusterError;
use crate::error::token_limit_message;
use crate::support::http_model_error;

impl OllamaClient {
    /// Streams the response as thinking/text deltas: Ollama sends
    /// NDJSON chunks with stream=true. No total timeout: a long
    /// generation must not be cut by one; the cancel token (Cluster)
    /// stops runaway streams.
    pub(super) async fn stream_impl(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        if prompt.history.is_empty() {
            self.stream_generate(prompt).await
        } else {
            self.stream_chat(prompt).await
        }
    }

    /// Streams /api/generate with stream=true.
    async fn stream_generate(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let url = format!("{}/api/generate", self.host);
        let request = self.generate_request(prompt, true);

        let response = self.client.post(&url).json(&request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await?;
            return Err(http_model_error(status, &body_text));
        }

        Ok(spawn_ollama_chunk_forwarder(
            response,
            false,
            token_limit_message(&self.model, self.params.num_predict),
        ))
    }

    /// Streams /api/chat with stream=true — same chunk flow as
    /// /api/generate, see `stream_generate`.
    async fn stream_chat(
        &self,
        prompt: &Prompt,
    ) -> Result<mpsc::Receiver<Result<PromptEvent, ClusterError>>, ClusterError> {
        let url = format!("{}/api/chat", self.host);
        let request = self.chat_request(prompt, true);

        let response = self.client.post(&url).json(&request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await?;
            return Err(http_model_error(status, &body_text));
        }

        Ok(spawn_ollama_chunk_forwarder(
            response,
            true,
            token_limit_message(&self.model, self.params.num_predict),
        ))
    }
}

/// Forwards Ollama NDJSON chunks into the event channel; the channel
/// closes on stream end or transport error. `chat` selects the chunk
/// shape: /api/chat nests thinking/content under "message".
fn spawn_ollama_chunk_forwarder(
    response: Response,
    chat: bool,
    token_limit_message: String,
) -> mpsc::Receiver<Result<PromptEvent, ClusterError>> {
    let (sender, receiver) = mpsc::channel(constants::STREAM_CHANNEL_SIZE);

    tokio::spawn(async move {
        let mut buffer = Vec::new();
        let mut stream = response.bytes_stream();
        let mut saw_text = false;

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(error) => {
                    log::error!("Ollama stream failed: {}", error);
                    let _ = sender.send(Err(ClusterError::Http(error))).await;
                    return;
                }
            };
            buffer.extend_from_slice(&bytes);

            // TCP chunks may split mid-line — decode only complete NDJSON lines.
            while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                if !forward_ollama_line(&sender, &line, chat, &token_limit_message, &mut saw_text)
                    .await
                {
                    return;
                }
            }
        }

        // A server may omit the final newline.
        if !buffer.is_empty() {
            let line = std::mem::take(&mut buffer);
            let _ = forward_ollama_line(&sender, &line, chat, &token_limit_message, &mut saw_text)
                .await;
        }
    });

    receiver
}

/// Forwards one complete NDJSON line; false requests stream termination.
/// `saw_text` tracks whether any non-empty text delta went out: the
/// output-token limit with nothing produced becomes a stream error,
/// mirroring the non-stream call paths.
async fn forward_ollama_line(
    sender: &mpsc::Sender<Result<PromptEvent, ClusterError>>,
    line: &[u8],
    chat: bool,
    token_limit_message: &str,
    saw_text: &mut bool,
) -> bool {
    let line = String::from_utf8_lossy(line);
    let Ok(json) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
        return true;
    };

    let (thinking, text) = if chat {
        (
            json["message"]["thinking"].as_str(),
            json["message"]["content"].as_str(),
        )
    } else {
        (json["thinking"].as_str(), json["response"].as_str())
    };

    if text.is_some_and(|text| !text.is_empty()) {
        *saw_text = true;
    }

    if json["done_reason"].as_str() == Some(constants::TOKEN_LIMIT_REASON) && !*saw_text {
        log::warn!("Ollama stream hit the output token limit");
        let _ = sender
            .send(Err(ClusterError::TokenLimitExceeded(
                token_limit_message.to_string(),
            )))
            .await;
        return false;
    }

    // empty deltas stay out: the final done chunk carries "" and is noise
    if let Some(text) = thinking.filter(|text| !text.is_empty()) {
        if sender
            .send(Ok(PromptEvent::Thinking(text.to_string())))
            .await
            .is_err()
        {
            return false;
        }
    }
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        if sender
            .send(Ok(PromptEvent::Text(text.to_string())))
            .await
            .is_err()
        {
            return false;
        }
    }

    // The counters ride the final done chunk.
    let prompt_eval_count = json["prompt_eval_count"].as_u64();
    let prompt_eval_cached = json["prompt_eval_cached_count"].as_u64();
    let eval_count = json["eval_count"].as_u64();
    if prompt_eval_count.is_some() || eval_count.is_some() {
        let facts = ollama_stats(prompt_eval_count, prompt_eval_cached, eval_count);
        if sender
            .send(Ok(PromptEvent::Usage {
                cached_tokens: facts.cached_tokens,
                input_tokens: facts.input_tokens,
                output_tokens: facts.output_tokens,
                thinking_tokens: facts.thinking_tokens,
            }))
            .await
            .is_err()
        {
            return false;
        }
    }

    true
}
