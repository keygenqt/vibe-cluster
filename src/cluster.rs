// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Model cluster for distributing prompts across multiple LLM providers.
//! Creates internal clients from connection parameters and handles
//! prompt distribution, retries, and progress reporting.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

use crate::cancel::CancelToken;
use crate::client::Client;
use crate::clients;
use crate::data::BatchOptions;
use crate::data::ConnectionParams;
use crate::data::Prompt;
use crate::data::PromptEvent;
use crate::data::PromptResult;
use crate::data::Provider;
use crate::error::ClusterError;

/// Model cluster that manages multiple LLM connections.
pub struct Cluster {
    clients: Vec<Arc<Mutex<Box<dyn Client>>>>,
}

impl Cluster {
    /// Create cluster from connection settings.
    /// Validates connections and initializes internal clients.
    pub fn new(connections: Vec<ConnectionParams>) -> Result<Self, ClusterError> {
        if connections.is_empty() {
            return Err(ClusterError::Config(
                "At least one connection required".into(),
            ));
        }

        let mut clients = Vec::new();
        for params in connections {
            let count = params.parallel.max(1);
            for _ in 0..count {
                let client = clients::create_client(params.clone());
                clients.push(Arc::new(Mutex::new(client)));
            }
        }

        Ok(Self { clients })
    }

    /// Checks that all clients are available. Returns error with details if any client is unreachable.
    pub async fn is_available(&self) -> Result<(), ClusterError> {
        for client in &self.clients {
            let client = client.lock().await;
            match client.is_available().await {
                Ok(true) => continue,
                Ok(false) => {
                    return Err(ClusterError::Unavailable(format!(
                        "{} ({:?}) is not available",
                        client.model_name(),
                        client.provider()
                    )));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Returns list of unique host URLs in this cluster.
    pub async fn hosts(&self) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();
        for client in &self.clients {
            let client = client.lock().await;
            let host = client.host().to_string();
            if !result.contains(&host) {
                result.push(host);
            }
        }
        result
    }

    /// Returns list of unique model names in this cluster.
    pub async fn models(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for client in &self.clients {
            let client = client.lock().await;
            let model = client.model_name().to_string();
            if !names.contains(&model) {
                names.push(model);
            }
        }
        names
    }

    /// Returns list of unique provider types in this cluster.
    pub async fn providers(&self) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();
        for client in &self.clients {
            let client = client.lock().await;
            let provider = format!("{:?}", client.provider());
            if !result.contains(&provider) {
                result.push(provider);
            }
        }
        result
    }

    /// Send a single prompt with default options.
    pub async fn call(&self, prompt: Prompt) -> Result<PromptResult, ClusterError> {
        self.call_with_options(prompt, &BatchOptions::default())
            .await
    }

    /// Send a single prompt with custom options.
    pub async fn call_with_options(
        &self,
        prompt: Prompt,
        options: &BatchOptions,
    ) -> Result<PromptResult, ClusterError> {
        let mut results: Vec<PromptResult> = self
            .batch_call_with_options(&[prompt], options, |_, _, _| {})
            .await?;
        results
            .pop()
            .ok_or(ClusterError::AllFailed("No response from cluster".into()))
    }

    /// Streams one prompt through the best available client (local
    /// models first, loaded models first). Thinking and text deltas go
    /// to `on_event` as they arrive, the aggregated `PromptResult`
    /// comes back. Cancellation cuts the stream: the result carries
    /// `ClusterError::Cancelled` and whatever arrived before the cut.
    /// One attempt — no retries.
    pub async fn call_stream<F>(
        &self,
        prompt: Prompt,
        cancel: &CancelToken,
        on_event: F,
    ) -> Result<PromptResult, ClusterError>
    where
        F: Fn(&PromptEvent) + Send + Sync + 'static,
    {
        // The same selection as the batch path: reachable, sorted, vision-filtered.
        let active_clients = self
            .prepare_clients(std::slice::from_ref(&prompt), &BatchOptions::default())
            .await?;
        let client = active_clients[0].clone();

        let (provider, model_name, host) = {
            let guard = client.lock().await;
            (
                guard.provider(),
                guard.model_name().to_string(),
                guard.host().to_string(),
            )
        };

        let start = Instant::now();
        let raw_cancel = cancel.raw();

        // Cancellation also covers connection and response-header waiting.
        let mut receiver = if cancel.is_cancelled() {
            None
        } else {
            let guard = client.lock().await;
            tokio::select! {
                _ = raw_cancel.cancelled() => None,
                receiver = guard.call_stream(&prompt) => Some(receiver?),
            }
        };

        let mut thinking = String::new();
        let mut text = String::new();
        let mut stream_error = None;

        if let Some(receiver) = receiver.as_mut() {
            let cancelled = raw_cancel.cancelled();
            tokio::pin!(cancelled);

            loop {
                tokio::select! {
                    _ = &mut cancelled => break,
                    event = receiver.recv() => match event {
                        Some(Ok(PromptEvent::Thinking(chunk))) => {
                            thinking.push_str(&chunk);
                            on_event(&PromptEvent::Thinking(chunk));
                        }
                        Some(Ok(PromptEvent::Text(chunk))) => {
                            text.push_str(&chunk);
                            on_event(&PromptEvent::Text(chunk));
                        }
                        Some(Err(error)) => {
                            stream_error = Some(error);
                            break;
                        }
                        None => break,
                    },
                }
            }
        }

        let error = if cancel.is_cancelled() {
            Some(ClusterError::Cancelled)
        } else {
            stream_error
        };

        Ok(PromptResult {
            prompt,
            provider,
            host,
            model: model_name,
            duration_ms: start.elapsed().as_millis() as u64,
            thinking: if thinking.is_empty() {
                None
            } else {
                Some(thinking)
            },
            text: if text.is_empty() { None } else { Some(text) },
            error,
        })
    }

    /// Send multiple prompts with default options.
    /// Calls `on_progress(result, current, total)` after each completed prompt.
    pub async fn batch_call<F>(
        &self,
        prompts: &[Prompt],
        on_progress: F,
    ) -> Result<Vec<PromptResult>, ClusterError>
    where
        F: Fn(&PromptResult, usize, usize) + Send + Sync + 'static,
    {
        self.batch_call_with_options(prompts, &BatchOptions::default(), on_progress)
            .await
    }

    /// Send multiple prompts with custom options.
    /// Calls `on_progress(result, current, total)` after each completed prompt.
    pub async fn batch_call_with_options<F>(
        &self,
        prompts: &[Prompt],
        options: &BatchOptions,
        on_progress: F,
    ) -> Result<Vec<PromptResult>, ClusterError>
    where
        F: Fn(&PromptResult, usize, usize) + Send + Sync + 'static,
    {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Prepare, filter, and prioritize available cluster nodes before execution
        let active_clients = self.prepare_clients(prompts, options).await?;

        let total = prompts.len();

        // Initialize the shared thread-safe buffer with lazy None values
        // using resize_with to avoid requiring the Clone trait on PromptResult.
        let mut vec_buffer = Vec::with_capacity(total);
        vec_buffer.resize_with(total, || None);
        let results = Arc::new(Mutex::new(vec_buffer));

        let next_index = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prompts = Arc::new(prompts.to_vec());
        let on_progress = Arc::new(on_progress);
        let mut handles = Vec::new();

        for client in active_clients {
            let client_arc = client.clone();
            let results = results.clone();
            let next_index = next_index.clone();
            let completed = completed.clone();
            let options = options.clone();
            let prompts = prompts.clone();
            let on_progress = on_progress.clone();

            let handle = tokio::spawn(async move {
                loop {
                    let idx = next_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }
                    let prompt = &prompts[idx];

                    // A cancelled token fills the remaining slots with
                    // Cancelled results instead of dropping them
                    if options.cancel.is_cancelled() {
                        let (provider, model_name, host) = {
                            let guard = client_arc.lock().await;
                            (
                                guard.provider(),
                                guard.model_name().to_string(),
                                guard.host().to_string(),
                            )
                        };
                        let result = PromptResult {
                            prompt: prompt.clone(),
                            provider,
                            host,
                            model: model_name,
                            duration_ms: 0,
                            thinking: None,
                            text: None,
                            error: Some(ClusterError::Cancelled),
                        };
                        let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        on_progress(&result, done, total);
                        results.lock().await[idx] = Some(result);
                        continue;
                    }

                    // Extract static metadata upfront to ensure it's available for error reports
                    let (provider, model_name, host) = {
                        let guard = client_arc.lock().await;
                        (
                            guard.provider(),
                            guard.model_name().to_string(),
                            guard.host().to_string(),
                        )
                    };

                    let result = if options.exclusive && provider == Provider::Ollama {
                        let eject_res = {
                            let guard = client_arc.lock().await;
                            guard.eject().await
                        };
                        if let Err(e) = eject_res {
                            PromptResult {
                                prompt: prompt.clone(),
                                provider,
                                host: host.clone(),
                                model: model_name.clone(),
                                duration_ms: 0,
                                thinking: None,
                                text: None,
                                error: Some(e),
                            }
                        } else {
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                            let guard = client_arc.lock().await;
                            match guard
                                .execute(prompt, options.retries.unwrap_or(1), &options.cancel)
                                .await
                            {
                                Ok(r) => r,
                                Err(e) => PromptResult {
                                    prompt: prompt.clone(),
                                    provider,
                                    host: host.clone(),
                                    model: model_name.clone(),
                                    duration_ms: 0,
                                    thinking: None,
                                    text: None,
                                    error: Some(e),
                                },
                            }
                        }
                    } else {
                        let guard = client_arc.lock().await;
                        match guard
                            .execute(prompt, options.retries.unwrap_or(1), &options.cancel)
                            .await
                        {
                            Ok(r) => r,
                            Err(e) => PromptResult {
                                prompt: prompt.clone(),
                                provider,
                                host: host.clone(),
                                model: model_name.clone(),
                                duration_ms: 0,
                                thinking: None,
                                text: None,
                                error: Some(e),
                            },
                        }
                    };

                    // Trigger progress callback using a clean borrow before moving ownership
                    let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    on_progress(&result, done, total);

                    // Transfer ownership of the result directly into the shared buffer without cloning
                    {
                        let mut res_guard = results.lock().await;
                        res_guard[idx] = Some(result);
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all worker handles to complete and propagate thread panics immediately
        for handle in handles {
            handle
                .await
                .map_err(|e| ClusterError::AllFailed(format!("Worker panicked: {}", e)))?;
        }

        let mut final_guard = results.lock().await;
        let mut final_results = Vec::with_capacity(total);

        // Fail-fast verification to catch critical thread execution drops or panics
        for (idx, r) in final_guard.iter_mut().enumerate() {
            if let Some(res) = r.take() {
                final_results.push(res);
            } else {
                return Err(ClusterError::AllFailed(format!(
                    "Fatal system error: Worker thread panicked, died, or dropped task at index {} prematurely.",
                    idx
                )));
            }
        }

        Ok(final_results)
    }

    /// Filters and sorts clients: reachable, local-first, active-first, vision-filtered.
    async fn prepare_clients(
        &self,
        prompts: &[Prompt],
        options: &BatchOptions,
    ) -> Result<Vec<Arc<Mutex<Box<dyn Client>>>>, ClusterError> {
        // A busy client still matches static filters; wait for its metadata.
        let mut active_clients = Vec::new();
        for client in &self.clients {
            if options.matches_client(client).await {
                active_clients.push(client.clone());
            }
        }

        // Check availability in parallel to avoid sequential network bottlenecks
        let mut tasks = Vec::new();
        for client in &active_clients {
            let client_clone = client.clone();
            tasks.push(tokio::spawn(async move {
                let has_access = client_clone
                    .lock()
                    .await
                    .is_available()
                    .await
                    .is_ok_and(|v| v);
                if has_access { Some(client_clone) } else { None }
            }));
        }

        // Collect parallel task results matching the exact source collection type
        let candidates = active_clients.len();
        let mut reachable = Vec::new();
        for task in futures::future::join_all(tasks).await {
            if let Ok(Some(client)) = task {
                reachable.push(client);
            }
        }
        let mut active_clients = reachable;

        log::debug!(
            "Cluster: {}/{} clients reachable",
            active_clients.len(),
            candidates
        );

        if active_clients.is_empty() {
            return Err(ClusterError::Unavailable(
                "No models are currently available.".into(),
            ));
        }

        // Gather metadata safely while minimizing mutex lock guard lifetime
        let mut with_meta: Vec<(_, bool, bool)> = Vec::new();
        for client in &active_clients {
            let (is_local, is_active) = {
                let guard = client.lock().await;
                let local = guard.provider() == Provider::Ollama;
                (local, guard.is_active().await.unwrap_or(false))
            };
            with_meta.push((client.clone(), is_local, is_active));
        }

        // Sort clients: local models first (Ollama), then active ones within the same group
        with_meta.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        active_clients = with_meta.into_iter().map(|(c, _, _)| c).collect();

        // Filter for vision capabilities if any prompt contains images
        let has_images = prompts
            .iter()
            .any(|p| p.images.as_ref().is_some_and(|images| !images.is_empty()));
        if has_images {
            let mut vision_clients = Vec::new();
            for client in &active_clients {
                let is_vision = client.lock().await.is_vision().await.unwrap_or(false);
                if is_vision {
                    vision_clients.push(client.clone());
                }
            }

            if vision_clients.is_empty() {
                return Err(ClusterError::VisionNotSupported(
                "No vision-capable models found in cluster. Add a vision model (e.g. minicpm-v) to your config.".into(),
            ));
            }
            active_clients = vision_clients;
        }

        Ok(active_clients)
    }
}
