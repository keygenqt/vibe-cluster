// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Connection validation.
//! See [`crate::validate`] module-level docs for the validation contract.

use crate::data::Connection;
use crate::data::Params;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::validate::ValidateTrait;

impl ValidateTrait for Connection {
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

        // The params variant must match the provider family.
        let family_matches = match (&self.provider, &self.params) {
            (Provider::Ollama, Params::Ollama(_)) => true,
            (
                Provider::OpenAI
                | Provider::DeepSeek
                | Provider::Qwen
                | Provider::Moonshot
                | Provider::Zhipu
                | Provider::Yandex,
                Params::OpenAi(_),
            ) => true,
            (Provider::GigaChat, Params::GigaChat(_)) => true,
            _ => false,
        };
        if !family_matches {
            return Err(ClusterError::Config(format!(
                "Connection '{}': params does not match the provider family.",
                self.model
            )));
        }

        // Ranges and limits: the generation ones live on the params,
        // the connection ones stay here.
        self.params.validate()?;
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

        // Yandex routes by the project folder: without it the model must
        // already be a full gpt:// URI, otherwise the provider rejects
        // the call at runtime.
        if self.provider == Provider::Yandex
            && self
                .project
                .as_ref()
                .map_or(true, |project| project.trim().is_empty())
            && !self.model.starts_with("gpt://")
        {
            return Err(ClusterError::Config(format!(
                "Connection '{}': project is required for Yandex unless the model is a gpt:// URI.",
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

impl ValidateTrait for Params {
    /// Checks the ranges of the set parameters: a `None` means the
    /// provider default applies and carries no range.
    fn validate(&self) -> Result<(), ClusterError> {
        match self {
            Params::Ollama(params) => {
                check_f64("temperature", params.temperature, 0.0, 2.0)?;
                check_f64("top_p", params.top_p, 0.0, 1.0)?;
                check_zero("top_k", params.top_k, 0)?;
                check_f64("min_p", params.min_p, 0.0, 1.0)?;
                check_stop(params.stop.as_ref())?;
                check_zero("num_ctx", params.num_ctx, 0)?;
                check_zero("num_predict", params.num_predict, 0)?;
            }
            Params::OpenAi(params) => {
                check_f64("temperature", params.temperature, 0.0, 2.0)?;
                check_f64("top_p", params.top_p, 0.0, 1.0)?;
                check_zero("top_k", params.top_k, 0)?;
                check_f64("min_p", params.min_p, 0.0, 1.0)?;
                check_stop(params.stop.as_ref())?;
                check_zero("num_predict", params.num_predict, 0)?;
            }
            Params::GigaChat(params) => {
                check_f64("temperature", params.temperature, 0.0, 2.0)?;
                check_f64("top_p", params.top_p, 0.0, 1.0)?;
                check_zero("num_predict", params.num_predict, 0)?;
            }
        }
        Ok(())
    }
}

/// Range check of an optional f64 parameter.
fn check_f64(name: &str, value: Option<f64>, min: f64, max: f64) -> Result<(), ClusterError> {
    if value.is_some_and(|value| !(min..=max).contains(&value)) {
        Err(ClusterError::Config(format!(
            "Generation parameters: {} must be {}-{}.",
            name, min, max
        )))
    } else {
        Ok(())
    }
}

/// Zero check of an optional numeric parameter.
fn check_zero<T: PartialEq + Copy>(
    name: &str,
    value: Option<T>,
    zero: T,
) -> Result<(), ClusterError> {
    if value.is_some_and(|value| value == zero) {
        Err(ClusterError::Config(format!(
            "Generation parameters: {} must be > 0.",
            name
        )))
    } else {
        Ok(())
    }
}

/// Stop sequences must not carry empty entries.
fn check_stop(stop: Option<&Vec<String>>) -> Result<(), ClusterError> {
    if stop.is_some_and(|stop| stop.iter().any(|item| item.trim().is_empty())) {
        Err(ClusterError::Config(
            "Generation parameters: stop sequences must not be empty.".into(),
        ))
    } else {
        Ok(())
    }
}
