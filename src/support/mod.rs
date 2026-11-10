// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Internal helpers used by the provider clients; not clients
//! themselves. Nothing here is re-exported from the root — the public
//! helpers live in `utils`.
//!
//! # Submodules
//!
//! - `auth` — process-wide cached OAuth access tokens.
//! - `http` — request building and the provider error extraction.
//! - `stream` — OpenAI-compatible SSE stream forwarding into PromptEvents.
//! - `usage` — the OpenAI-compatible `usage` node parsing.

mod auth;
mod http;
mod stream;
mod usage;

pub(crate) use auth::cached_token;
pub(crate) use auth::invalidate_token;
pub(crate) use auth::store_token;
pub(crate) use http::build_messages;
pub(crate) use http::http_model_error;
pub(crate) use stream::spawn_sse_forwarder;
pub(crate) use usage::parse_usage;
