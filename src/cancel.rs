// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Cancellation handle for in-flight requests.

use tokio_util::sync::CancellationToken;

/// A cancellation handle: clones of one token can be bound to several
/// requests, the owner cancels them all with a single `cancel()`.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    token: CancellationToken,
}

impl CancelToken {
    /// Creates a new token.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    /// Cancels every request bound to this token.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Returns true if the token is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// The raw token for `select!` inside the crate; never public.
    pub(crate) fn raw(&self) -> CancellationToken {
        self.token.clone()
    }
}
