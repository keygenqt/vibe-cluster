// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Live capability detection: /api/show facts, the vision tiers
//! (families → projector) as the older-servers fallback, the name
//! rules for everything the server did not report.

use super::OllamaClient;
use crate::constants;
use crate::data::Abilities;
use crate::data::Capabilities;
use crate::data::Input;
use crate::data::Output;
use crate::data::Provider;
use crate::error::ClusterError;

impl OllamaClient {
    /// Fetches /api/show for a model and returns the parsed JSON body.
    pub(super) async fn show_model(&self, name: &str) -> Result<serde_json::Value, ClusterError> {
        let url = format!("{}/api/show", self.host);
        let request = serde_json::json!({ "name": name });
        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(constants::CHECK_TIMEOUT)
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(ClusterError::Model(format!(
                "Ollama /api/show returned status: {}",
                response.status()
            )));
        }
        Ok(response.json().await?)
    }

    /// Maps a /api/show body into capabilities: reported keys fill the
    /// matching fields, the rest fall back to the vision tiers (families
    /// → projector) and the name rules.
    pub(super) fn show_capabilities(&self, show: &serde_json::Value, name: &str) -> Capabilities {
        let caps = show["capabilities"].as_array();
        let has = |cap: &str| caps.map(|c| c.iter().any(|v| v.as_str() == Some(cap)));

        let mut capabilities = Capabilities {
            input: Input {
                text: None,
                vision: has("vision"),
                audio: has("audio"),
                video: has("video"),
            },
            output: Output::default(),
            abilities: Abilities {
                tools: has("tools"),
                thinking: has("thinking"),
                embedding: has("embedding"),
                decision: has("decision"),
            },
        };

        // Older servers: vision models carry a CLIP encoder, a dedicated
        // projector, or a known vision family.
        if capabilities.input.vision.is_none() {
            let families = show["details"]["families"].as_array();
            let has_vision_family = families
                .map(|f| {
                    f.iter().any(|v| {
                        matches!(
                            v.as_str(),
                            Some("clip") | Some("mllama") | Some("qwen2vl") | Some("qwen3vl")
                        )
                    })
                })
                .unwrap_or(false);
            let has_projector = show["projector_info"].is_object();
            if has_vision_family || has_projector {
                capabilities.input.vision = Some(true);
            }
        }

        // Fields the server did not report fall back to the name rules.
        let fallback = self.rules.capabilities(&Provider::Ollama, name);
        let fill = |target: &mut Option<bool>, source: Option<bool>| {
            if target.is_none() {
                *target = source;
            }
        };
        fill(&mut capabilities.input.text, fallback.input.text);
        fill(&mut capabilities.input.vision, fallback.input.vision);
        fill(&mut capabilities.input.audio, fallback.input.audio);
        fill(&mut capabilities.input.video, fallback.input.video);
        fill(&mut capabilities.output.text, fallback.output.text);
        fill(&mut capabilities.output.image, fallback.output.image);
        fill(&mut capabilities.output.audio, fallback.output.audio);
        fill(&mut capabilities.output.video, fallback.output.video);
        fill(&mut capabilities.abilities.tools, fallback.abilities.tools);
        fill(
            &mut capabilities.abilities.thinking,
            fallback.abilities.thinking,
        );
        fill(
            &mut capabilities.abilities.embedding,
            fallback.abilities.embedding,
        );
        fill(
            &mut capabilities.abilities.decision,
            fallback.abilities.decision,
        );

        capabilities
    }
}
