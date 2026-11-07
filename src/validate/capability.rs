// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! CapabilityRules file validation.
//! See [`crate::validate`] module-level docs for the validation contract.

use regex::Regex;

use crate::capability::RawField;
use crate::capability::RawFile;
use crate::capability::SUPPORTED_VERSION;
use crate::capability::parse_capability;
use crate::capability::parse_provider;
use crate::error::ClusterError;
use crate::validate::ValidateTrait;

impl ValidateTrait for RawFile {
    /// Checks the rules file structure: version, provider names,
    /// capability keys, regex syntax.
    fn validate(&self) -> Result<(), ClusterError> {
        if self.version != SUPPORTED_VERSION {
            return Err(ClusterError::Config(format!(
                "Capability rules: unsupported version {}, expected {}",
                self.version, SUPPORTED_VERSION
            )));
        }
        for (provider, raw) in &self.providers {
            if parse_provider(provider).is_none() {
                return Err(ClusterError::Config(format!(
                    "Capability rules: unknown provider '{}'",
                    provider
                )));
            }
            for (group, capabilities) in raw.groups() {
                for (key, field) in capabilities {
                    if parse_capability(group, key).is_none() {
                        return Err(ClusterError::Config(format!(
                            "Capability rules: unknown capability '{}.{}' for provider '{}'",
                            group, key, provider
                        )));
                    }
                    validate_field(field, &format!("providers.{}.{}.{}", provider, group, key))?;
                }
            }
        }
        Ok(())
    }
}

/// Validates one capability rule set: no empty plain entries, regexes compile.
fn validate_field(field: &RawField, path: &str) -> Result<(), ClusterError> {
    for entry in field.names.iter().chain(&field.keywords) {
        if entry.trim().is_empty() {
            return Err(ClusterError::Config(format!(
                "Capability rules: empty entry in '{}'",
                path
            )));
        }
    }
    for pattern in field.include.iter().chain(&field.exclude) {
        Regex::new(pattern).map_err(|error| {
            ClusterError::Config(format!(
                "Capability rules: invalid regex '{}' in '{}': {}",
                pattern, path, error
            ))
        })?;
    }
    Ok(())
}
