// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Request building and provider error extraction shared by the
//! OpenAI-compatible clients (OpenAI itself, GigaChat).

use crate::data::Prompt;
use crate::error::ClusterError;
use crate::utils::mime_from_base64;

/// One OpenAI-compatible `content` value: a plain string, or the
/// text+images part array when images ride along.
pub(crate) fn content_value(text: &str, images: &[String]) -> serde_json::Value {
    if images.is_empty() {
        return serde_json::Value::String(text.to_string());
    }
    let mut content: Vec<serde_json::Value> =
        vec![serde_json::json!({ "type": "text", "text": text })];
    for img in images {
        // Prompt.images accepts raw base64 or a ready-made data: URI;
        // each client normalizes the payload for its provider.
        let url = if img.starts_with("data:") {
            img.clone()
        } else {
            format!("data:{};base64,{}", mime_from_base64(img), img)
        };
        content.push(serde_json::json!({
            "type": "image_url",
            "image_url": {"url": url}
        }));
    }
    serde_json::Value::Array(content)
}

/// The OpenAI-compatible `messages` array from a prompt: the system
/// prompt, the history turns, the current user message; images ride
/// as data URIs.
pub(crate) fn build_messages(prompt: &Prompt) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();

    if let Some(system) = &prompt.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }

    for message in &prompt.history {
        messages.push(serde_json::json!({
            "role": message.role,
            "content": content_value(&message.content, &message.images),
        }));
    }

    messages.push(serde_json::json!({
        "role": "user",
        "content": content_value(&prompt.user, prompt.images.as_deref().unwrap_or(&[])),
    }));

    messages
}

/// Builds a model error from a failed HTTP response body, digging the
/// error message out of the known JSON envelopes: the standard nested
/// `error.message`, a flat `error` string, and an `error` string that
/// itself carries a nested JSON envelope (Ollama-style).
pub(crate) fn http_model_error(status: reqwest::StatusCode, body_text: &str) -> ClusterError {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_text) {
        if let Some(msg) = json["error"]["message"].as_str() {
            return ClusterError::HttpStatus {
                status: status.as_u16(),
                message: msg.to_string(),
            };
        }
        if let Some(payload) = json["error"].as_str() {
            if let Ok(inner) = serde_json::from_str::<serde_json::Value>(payload) {
                if let Some(msg) = inner["error"]["message"].as_str() {
                    return ClusterError::HttpStatus {
                        status: status.as_u16(),
                        message: msg.to_string(),
                    };
                }
            }
            return ClusterError::HttpStatus {
                status: status.as_u16(),
                message: payload.to_string(),
            };
        }
    }
    ClusterError::HttpStatus {
        status: status.as_u16(),
        message: body_text.to_string(),
    }
}
