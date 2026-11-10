// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Cross-client mechanics constants: timeouts, backoffs, channel
//! sizes, protocol values. Everything here is `pub(crate)` — the
//! public API carries no constants. Provider-specific values
//! (`DEFAULT_API_HOST`, `OAUTH_URL`, `TOKEN_LIFETIME`) stay in the
//! provider's own files.

use std::time::Duration;

/// Timeout of the short endpoint calls: the checks (/models, /api/tags,
/// /api/ps, /api/show) and the Ollama eject.
pub(crate) const CHECK_TIMEOUT: Duration = Duration::from_secs(5);

/// Timeout of the OAuth token request.
pub(crate) const OAUTH_TIMEOUT: Duration = Duration::from_secs(10);

/// Default of `Connection::timeout_secs` — the full call timeout.
pub(crate) const DEFAULT_CALL_TIMEOUT_SECS: u64 = 120;

/// Pause between prompt retries.
pub(crate) const RETRY_BACKOFF: Duration = Duration::from_millis(500);

/// Pause after eject before the next request — the driver needs a moment to release VRAM.
pub(crate) const EJECT_SETTLE: Duration = Duration::from_millis(500);

/// Event channel capacity of the stream forwarders.
pub(crate) const STREAM_CHANNEL_SIZE: usize = 16;

/// Server statuses that trigger an automatic retry.
pub(crate) const RETRYABLE_STATUS_RANGE: std::ops::RangeInclusive<u16> = 500..=599;

/// The access token went stale — refresh and retry.
pub(crate) const STATUS_UNAUTHORIZED: u16 = 401;

/// Bearer auth scheme shared by the OpenAI-compatible clients.
pub(crate) const AUTH_BEARER: &str = "Bearer";

/// The finish/done reason meaning the output token limit was hit.
pub(crate) const TOKEN_LIMIT_REASON: &str = "length";

/// Default of `BatchOptions::retries`.
pub(crate) const DEFAULT_RETRIES: usize = 5;
