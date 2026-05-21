//! Model cluster for distributing prompts across multiple LLM providers.
//! Creates internal clients from connection parameters and handles
//! prompt distribution, retries, and progress reporting.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::clients::client::Client;
use crate::error::ClusterError;
use crate::options::BatchOptions;
use crate::params::ConnectionParams;
use crate::provider::Provider;
use crate::result::PromptResult;
use crate::{Prompt, clients};

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

        // Filter clients by options
        let mut active_clients: Vec<_> = self
            .clients
            .iter()
            .filter(|c| options.matches_client(c))
            .cloned()
            .collect();

        // Exclude clients with unreachable servers
        let mut reachable = Vec::new();
        for client in &active_clients {
            if client.lock().await.is_available().await.is_ok_and(|v| v) {
                reachable.push(client.clone());
            }
        }
        active_clients = reachable;

        if active_clients.is_empty() {
            return Err(ClusterError::Config("No matching models available".into()));
        }

        // Sort: local models first (Ollama > cloud), then active first within each group
        let mut with_meta: Vec<(_, bool, bool)> = Vec::new();
        for client in &active_clients {
            let guard = client.lock().await;
            let is_local = guard.provider() == Provider::Ollama;
            let is_active = guard.is_active().await.unwrap_or(false);
            with_meta.push((client.clone(), is_local, is_active));
        }
        with_meta.sort_by(|a, b| {
            b.1.cmp(&a.1) // local first (Ollama > cloud)
                .then(b.2.cmp(&a.2)) // active first within same type
        });
        active_clients = with_meta.into_iter().map(|(c, _, _)| c).collect();

        let total = prompts.len();
        let results = Arc::new(Mutex::new(vec![Option::<PromptResult>::None; total]));
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
                                host: host,
                                model: model_name,
                                duration_ms: 0,
                                text: None,
                                error: Some(e),
                            }
                        } else {
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

                            let guard = client_arc.lock().await;
                            match guard.execute(prompt, options.retries.unwrap_or(1)).await {
                                Ok(r) => r,
                                Err(e) => PromptResult {
                                    prompt: prompt.clone(),
                                    provider,
                                    host: host,
                                    model: model_name,
                                    duration_ms: 0,
                                    text: None,
                                    error: Some(e),
                                },
                            }
                        }
                    } else {
                        let guard = client_arc.lock().await;
                        match guard.execute(prompt, options.retries.unwrap_or(1)).await {
                            Ok(r) => r,
                            Err(e) => PromptResult {
                                prompt: prompt.clone(),
                                provider,
                                host: host,
                                model: model_name,
                                duration_ms: 0,
                                text: None,
                                error: Some(e),
                            },
                        }
                    };

                    {
                        let mut res_guard = results.lock().await;
                        res_guard[idx] = Some(result.clone());
                    }

                    let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    on_progress(&result, done, total);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle
                .await
                .map_err(|e| ClusterError::AllFailed(format!("Worker panicked: {}", e)))?;
        }

        let mut final_guard = results.lock().await;
        let results: Vec<PromptResult> = final_guard
            .iter_mut()
            .map(|r| {
                r.take().unwrap_or(PromptResult {
                    prompt: Prompt::new(""),
                    provider: Provider::Ollama,
                    host: String::new(),
                    model: String::new(),
                    duration_ms: 0,
                    text: None,
                    error: Some(ClusterError::AllFailed(
                        "No worker picked up this prompt".into(),
                    )),
                })
            })
            .collect();

        Ok(results)
    }
}
