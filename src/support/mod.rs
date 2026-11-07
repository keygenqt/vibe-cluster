// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers used by the provider clients; not clients themselves.
//!
//! # Submodules
//!
//! - `image` — image payload normalization: MIME detection, data URI stripping.
//! - `sse` — OpenAI-compatible SSE stream forwarding into PromptEvents.

mod image;
mod sse;

pub(crate) use image::content_value;
pub use image::mime_from_base64;
pub use image::strip_data_uri;
pub(crate) use sse::spawn_sse_chunk_forwarder;
