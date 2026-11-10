// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible SSE stream parsing shared by the clients.

use futures_util::StreamExt;

use super::usage::parse_usage;
use reqwest::Response;
use tokio::sync::mpsc;

use crate::constants;
use crate::data::PromptEvent;
use crate::error::ClusterError;

/// Forwards SSE deltas into the event channel; the channel closes on
/// stream end (`data: [DONE]`) or transport error.
pub(crate) fn spawn_sse_forwarder(
    response: Response,
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
                    log::error!("SSE stream failed: {}", error);
                    let _ = sender.send(Err(ClusterError::Http(error))).await;
                    return;
                }
            };
            buffer.extend_from_slice(&bytes);

            // SSE events are separated by a blank line; decode only complete events.
            while let Some(end) = find_event_end(&buffer) {
                let event: Vec<u8> = buffer.drain(..end).collect();
                if !forward_event(&sender, &event, &token_limit_message, &mut saw_text).await {
                    return;
                }
            }
        }

        // A server may omit the final blank line.
        if !buffer.is_empty() {
            let event = std::mem::take(&mut buffer);
            let _ = forward_event(&sender, &event, &token_limit_message, &mut saw_text).await;
        }
    });

    receiver
}

/// Returns the end of the first SSE event, accepting LF and CRLF lines.
fn find_event_end(buffer: &[u8]) -> Option<usize> {
    for (index, byte) in buffer.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }
        match buffer.get(index + 1) {
            Some(b'\n') => return Some(index + 2),
            Some(b'\r') if buffer.get(index + 2) == Some(&b'\n') => {
                return Some(index + 3);
            }
            _ => {}
        }
    }
    None
}

/// Forwards one complete SSE event; false requests stream termination.
/// `saw_text` tracks whether any non-empty text delta went out: the
/// output-token limit with nothing produced becomes a stream error,
/// mirroring the non-stream call paths.
async fn forward_event(
    sender: &mpsc::Sender<Result<PromptEvent, ClusterError>>,
    event: &[u8],
    token_limit_message: &str,
    saw_text: &mut bool,
) -> bool {
    let event = String::from_utf8_lossy(event);

    for line in event.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            return false;
        }
        let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };

        // DeepSeek-R1 reports reasoning_content, OpenAI o-series — reasoning
        let delta = &json["choices"][0]["delta"];
        let thinking = delta["reasoning_content"]
            .as_str()
            .or_else(|| delta["reasoning"].as_str());
        let text = delta["content"].as_str();

        if text.is_some_and(|text| !text.is_empty()) {
            *saw_text = true;
        }

        if json["choices"][0]["finish_reason"].as_str() == Some(constants::TOKEN_LIMIT_REASON)
            && !*saw_text
        {
            log::warn!("SSE stream hit the output token limit");
            let _ = sender
                .send(Err(ClusterError::TokenLimitExceeded(
                    token_limit_message.to_string(),
                )))
                .await;
            return false;
        }

        if let Some(text) = thinking {
            if sender
                .send(Ok(PromptEvent::Thinking(text.to_string())))
                .await
                .is_err()
            {
                return false;
            }
        }
        if let Some(text) = text {
            if sender
                .send(Ok(PromptEvent::Text(text.to_string())))
                .await
                .is_err()
            {
                return false;
            }
        }

        // The usage rides the final chunk (OpenAI/Qwen/Yandex send it
        // in a separate empty-choices one); repeated reports are fine —
        // the consumer keeps the last one.
        let usage = &json["usage"];
        if usage.is_object() {
            let facts = parse_usage(usage);
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
    }

    true
}
