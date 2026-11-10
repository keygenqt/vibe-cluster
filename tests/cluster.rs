// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Live cluster integration tests. Requires a local Ollama with the
//! `qwen2.5-coder:3b-instruct` and `gemma4:12b` models pulled.
//!
//! cargo test --test cluster -- --test-threads=1 --nocapture

mod common;

use vibe_cluster::BatchOptions;
use vibe_cluster::CancelToken;
use vibe_cluster::Cluster;
use vibe_cluster::ClusterError;
use vibe_cluster::Connection;
use vibe_cluster::ModelFilter;
use vibe_cluster::OllamaParams;
use vibe_cluster::OpenAiParams;
use vibe_cluster::Params;
use vibe_cluster::Prompt;
use vibe_cluster::Provider;

use crate::common::ollama_connection::test_config;
use crate::common::ollama_connection::test_config_two_models;

#[tokio::test]
async fn test_single_call() {
    let cluster = Cluster::new(test_config()).unwrap();
    let result = cluster
        .call(Prompt::new("Reply with just: ok", None, None))
        .await
        .unwrap();
    assert!(result.error.is_none());
    assert!(result.text.unwrap().to_lowercase().contains("ok"));
}

#[tokio::test]
async fn test_batch_call_returns_in_order() {
    let cluster = Cluster::new(test_config()).unwrap();
    let prompts: Vec<Prompt> = vec![
        Prompt::new("Reply with just: one", None, None),
        Prompt::new("Reply with just: two", None, None),
        Prompt::new("Reply with just: three", None, None),
    ];
    let results = cluster.batch_call(&prompts, |_, _, _| {}).await.unwrap();

    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        assert!(r.error.is_none(), "prompt {} failed: {:?}", i, r.error);
    }
    assert!(results[0].text.as_ref().unwrap().contains("one"));
    assert!(results[1].text.as_ref().unwrap().contains("two"));
    assert!(results[2].text.as_ref().unwrap().contains("three"));
}

#[tokio::test]
async fn test_batch_call_parallel() {
    let cluster = Cluster::new(test_config_two_models()).unwrap();
    let prompts: Vec<Prompt> = (0..6)
        .map(|i| Prompt::new(format!("Reply with just: {}", i), None, None))
        .collect();
    let results = cluster.batch_call(&prompts, |_, _, _| {}).await.unwrap();

    assert_eq!(results.len(), 6);
    for (i, r) in results.iter().enumerate() {
        assert!(r.error.is_none(), "prompt {} failed: {:?}", i, r.error);
    }
}

#[tokio::test]
async fn test_local_only_filter() {
    let cluster = Cluster::new(test_config()).unwrap();
    let options = BatchOptions {
        models: ModelFilter::LocalOnly,
        ..Default::default()
    };
    let result = cluster
        .call_with_options(Prompt::new("Reply with just: local", None, None), &options)
        .await
        .unwrap();
    assert!(result.error.is_none());
    assert!(result.text.unwrap().to_lowercase().contains("local"));
}

#[tokio::test]
async fn test_cloud_only_filter_no_match() {
    let cluster = Cluster::new(test_config()).unwrap();
    let options = BatchOptions {
        models: ModelFilter::CloudOnly,
        ..Default::default()
    };
    let result = cluster
        .call_with_options(Prompt::new("Reply with just: test", None, None), &options)
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_exclusive_mode() {
    let cluster = Cluster::new(test_config()).unwrap();
    let options = BatchOptions {
        exclusive: true,
        ..Default::default()
    };
    let result = cluster
        .call_with_options(
            Prompt::new("Reply with just: exclusive", None, None),
            &options,
        )
        .await
        .unwrap();
    assert!(result.error.is_none());
    assert!(result.text.unwrap().to_lowercase().contains("exclusive"));
}

#[tokio::test]
async fn test_progress_callback() {
    let cluster = Cluster::new(test_config()).unwrap();
    let prompts: Vec<Prompt> = vec![
        Prompt::new("Reply with just: a", None, None),
        Prompt::new("Reply with just: b", None, None),
        Prompt::new("Reply with just: c", None, None),
    ];

    cluster
        .batch_call(&prompts, |result, done, total| {
            println!(
                "[{}/{}] {} in {}ms — {}",
                done,
                total,
                result.model,
                result
                    .stats
                    .as_ref()
                    .map(|stats| stats.duration_ms)
                    .unwrap_or(0),
                if result.error.is_none() { "OK" } else { "FAIL" }
            );
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn test_empty_prompts() {
    let cluster = Cluster::new(test_config()).unwrap();
    let results = cluster.batch_call(&[], |_, _, _| {}).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_empty_connections() {
    let result = Cluster::new(vec![]);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_unreachable_host() {
    let config = vec![Connection {
        provider: Provider::Ollama,
        host: "http://127.0.0.1:1".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(1),
        ..Default::default()
    }];
    let cluster = Cluster::new(config).unwrap();
    let result = cluster.call(Prompt::new("test", None, None)).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_batch_cancel() {
    let cluster = Cluster::new(test_config()).unwrap();

    let cancel = CancelToken::new();
    cancel.cancel();

    let options = BatchOptions {
        cancel,
        ..Default::default()
    };

    let prompts: Vec<Prompt> = (0..3)
        .map(|i| Prompt::new(format!("Reply with just: {}", i), None, None))
        .collect();

    let results = cluster
        .batch_call_with_options(&prompts, &options, |_, _, _| {})
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        assert!(
            matches!(r.error, Some(ClusterError::Cancelled)),
            "prompt {} expected Cancelled: {:?}",
            i,
            r.error
        );
    }
}

#[tokio::test]
async fn test_params_family_mismatch() {
    let config = vec![Connection {
        provider: Provider::GigaChat,
        host: "https://api.giga.chat/v1".into(),
        model: "GigaChat".into(),
        params: Params::Ollama(OllamaParams::default()),
        api_key: Some("key".into()),
        ..Default::default()
    }];

    // the params variant must follow the provider family
    let error = match Cluster::new(config) {
        Err(error) => error,
        Ok(_) => panic!("expected the params family mismatch"),
    };
    assert!(matches!(error, ClusterError::Config(_)));
}

#[tokio::test]
async fn test_yandex_requires_project() {
    let config = vec![Connection {
        provider: Provider::Yandex,
        host: "https://yandex.api/v1".into(),
        model: "yandexgpt-5-lite".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        api_key: Some("key".into()),
        ..Default::default()
    }];

    // without a folder id the call would fail at runtime — catch it at config
    let error = match Cluster::new(config) {
        Err(error) => error,
        Ok(_) => panic!("expected the missing project error"),
    };
    assert!(matches!(error, ClusterError::Config(_)));
}
