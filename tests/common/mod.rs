// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the integration tests.
//! Each test binary uses its own subset — unused helpers are fine here.

#![allow(dead_code)]

pub(crate) mod mock_server;
pub(crate) mod ollama_connection;
