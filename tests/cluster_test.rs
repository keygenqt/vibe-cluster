//! Cluster integration tests
//!
//! cargo test --test cluster_test -- --test-threads=1 --nocapture

use vibe_cluster::{BatchOptions, Cluster, ConnectionParams, ModelFilter, Prompt, Provider};

fn test_config() -> Vec<ConnectionParams> {
    vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(60),
        temperature: Some(0.1),
        seed: Some(42),
        num_ctx: Some(4096),
        num_predict: Some(2048),
        ..Default::default()
    }]
}

fn test_config_two_models() -> Vec<ConnectionParams> {
    vec![
        ConnectionParams {
            provider: Provider::Ollama,
            host: "http://localhost:11434".into(),
            model: "qwen2.5-coder:3b-instruct".into(),
            timeout_secs: Some(60),
            temperature: Some(0.1),
            seed: Some(42),
            num_ctx: Some(4096),
            num_predict: Some(2048),
            ..Default::default()
        },
        ConnectionParams {
            provider: Provider::Ollama,
            host: "http://localhost:11434".into(),
            model: "qwen2.5-coder:14b-instruct".into(),
            timeout_secs: Some(60),
            temperature: Some(0.1),
            seed: Some(42),
            num_ctx: Some(4096),
            num_predict: Some(2048),
            ..Default::default()
        },
    ]
}

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
                result.duration_ms,
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
    let config = vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:99999".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(1),
        ..Default::default()
    }];
    let cluster = Cluster::new(config).unwrap();
    let result = cluster.call(Prompt::new("test", None, None)).await;
    assert!(result.is_err());
}
