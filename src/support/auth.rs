// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Process-wide OAuth token cache shared across client instances.
//! Keyed by the caller's cache key (e.g. the auth key) — several auth
//! sources can coexist.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use tokio::sync::Mutex;

/// Process-wide token cache: cache key → (token, obtained at).
static TOKENS: LazyLock<Mutex<HashMap<String, (String, Instant)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns the cached token if it is younger than `lifetime`.
pub(crate) async fn cached_token(cache_key: &str, lifetime: Duration) -> Option<String> {
    let cache = TOKENS.lock().await;
    cache
        .get(cache_key)
        .filter(|(_, obtained)| obtained.elapsed() < lifetime)
        .map(|(token, _)| token.clone())
}

/// Stores a fresh token under the cache key.
pub(crate) async fn store_token(cache_key: &str, token: &str) {
    TOKENS
        .lock()
        .await
        .insert(cache_key.to_string(), (token.to_string(), Instant::now()));
}

/// Drops the cached token so the next call fetches a fresh one.
pub(crate) async fn invalidate_token(cache_key: &str) {
    TOKENS.lock().await.remove(cache_key);
}
