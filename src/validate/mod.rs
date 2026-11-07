// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Validation trait and contract.
//!
//! # Validation contract
//!
//! Every model implements [`ValidateTrait`] with a single
//! `validate() -> Result<(), ClusterError>` method. Validation runs
//! after parsing, before the parsed data is applied or executed; a
//! failure aborts the operation with a `ClusterError::Config` — no
//! invalid data reaches the engine.
//!
//! # Submodules
//!
//! - `capability` — `CapabilityRules` files: version, provider names,
//!   capability keys, regex syntax.
//! - `params` — connection: non-empty model; host is a valid http/https
//!   URL (empty allowed for providers with a built-in default);
//!   temperature/limits ranges; `api_key` required for the keyed clouds.

use crate::error::ClusterError;

/// Trait for types that can be validated.
pub trait ValidateTrait {
    fn validate(&self) -> Result<(), ClusterError>;
}

mod capability;
mod params;
