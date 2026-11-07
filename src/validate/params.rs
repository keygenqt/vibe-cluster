// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! ConnectionParams validation.
//! See [`crate::validate`] module-level docs for the validation contract.

use crate::data::ConnectionParams;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::validate::ValidateTrait;

impl ValidateTrait for ConnectionParams {
    /// Validate a single connection.
    fn validate(&self) -> Result<(), ClusterError> {
        let is_blank = |s: &str| s.trim().is_empty();

        // Required fields must not be empty.
        if is_blank(&self.model) {
            return Err(ClusterError::Config("Connection: model is empty.".into()));
        }

        // An empty host is legal only for providers with a built-in default.
        if is_blank(&self.host) && !matches!(self.provider, Provider::OpenAI | Provider::GigaChat) {
            return Err(ClusterError::Config(format!(
                "Connection '{}': host is empty.",
                self.model
            )));
        }
        if !is_blank(&self.host) {
            let url = reqwest::Url::parse(&self.host).map_err(|error| {
                ClusterError::Config(format!(
                    "Connection '{}': invalid host URL '{}': {}",
                    self.model, self.host, error
                ))
            })?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err(ClusterError::Config(format!(
                    "Connection '{}': host '{}' must be http/https.",
                    self.model, self.host
                )));
            }
        }

        // Ranges and limits.
        if self
            .temperature
            .is_some_and(|temperature| !(0.0..=2.0).contains(&temperature))
        {
            return Err(ClusterError::Config(format!(
                "Connection '{}': temperature must be 0.0-2.0.",
                self.model
            )));
        }
        if self.num_ctx.is_some_and(|value| value == 0) {
            return Err(ClusterError::Config(format!(
                "Connection '{}': num_ctx must be > 0.",
                self.model
            )));
        }
        if self.num_predict.is_some_and(|value| value == 0) {
            return Err(ClusterError::Config(format!(
                "Connection '{}': num_predict must be > 0.",
                self.model
            )));
        }
        if self.timeout_secs.is_some_and(|value| value == 0) {
            return Err(ClusterError::Config(format!(
                "Connection '{}': timeout_secs must be > 0.",
                self.model
            )));
        }
        if self.parallel == 0 {
            return Err(ClusterError::Config(format!(
                "Connection '{}': parallel must be > 0.",
                self.model
            )));
        }

        // Keyed clouds cannot work without a key; the OpenAI kind doubles
        // as a keyless gateway host (vLLM), so it stays lenient.
        if matches!(
            self.provider,
            Provider::DeepSeek
                | Provider::Qwen
                | Provider::Moonshot
                | Provider::Zhipu
                | Provider::Yandex
                | Provider::GigaChat
        ) && self
            .api_key
            .as_ref()
            .map_or(true, |key| key.trim().is_empty())
        {
            return Err(ClusterError::Config(format!(
                "Connection '{}': api_key is required for {:?}.",
                self.model, self.provider
            )));
        }

        Ok(())
    }
}
