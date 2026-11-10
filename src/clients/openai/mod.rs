// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible API client: OpenAI itself, DeepSeek, Qwen,
//! Moonshot, Zhipu, Yandex and any other provider with an
//! OpenAI-compatible API.
//!
//! # Submodules
//!
//! - `client` — the `OpenAiClient` entity and its `Client` impl.
//! - `requests` — request bodies, generation params, auth and model URIs.

mod call;
mod client;

pub use client::OpenAiClient;
