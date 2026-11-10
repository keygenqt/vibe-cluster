// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! GigaChat API client (Sber).
//! OAuth access token from Sber (30 min TTL, cached), then
//! OpenAI-compatible chat completions on api.giga.chat.
//!
//! # Submodules
//!
//! - `client` — the `GigaChatClient` entity and its `Client` impl.
//! - `requests` — OAuth token fetch, call/stream paths, generation params.

mod call;
mod client;

pub use client::GigaChatClient;
