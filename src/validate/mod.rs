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
//! - `capability` — `CapabilityRules` files: the file structure and
//!   the rule entries are valid.
//! - `params` — `Connection` and generation parameters: required
//!   fields are set and the values are in valid ranges.

use crate::error::ClusterError;

/// Trait for types that can be validated.
pub trait ValidateTrait {
    fn validate(&self) -> Result<(), ClusterError>;
}

mod capability;
mod params;
