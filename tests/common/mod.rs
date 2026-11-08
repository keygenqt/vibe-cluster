// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the integration tests.
//! Each test binary uses its own subset — unused helpers are fine here.

#![allow(dead_code)]

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use vibe_cluster::ConnectionParams;
use vibe_cluster::Provider;

pub async fn mock_server(
    status: &'static str,
    content_type: &'static str,
    chunks: Vec<Vec<u8>>,
) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 1024];

        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0, "connection closed before request headers");
            request.extend_from_slice(&buffer[..read]);
        }

        let body_len: usize = chunks.iter().map(Vec::len).sum();
        let headers = format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status, content_type, body_len
        );
        socket.write_all(headers.as_bytes()).await.unwrap();

        for chunk in chunks {
            socket.write_all(&chunk).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        socket.shutdown().await.unwrap();
    });

    (format!("http://{}", address), handle)
}

pub fn test_config() -> Vec<ConnectionParams> {
    vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(60),
        temperature: Some(0.1),
        seed: Some(42),
        num_ctx: Some(4096),
        num_predict: Some(2048),
        stats: true,
        ..Default::default()
    }]
}

pub fn test_config_two_models() -> Vec<ConnectionParams> {
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
            model: "gemma4:12b".into(),
            timeout_secs: Some(60),
            temperature: Some(0.1),
            seed: Some(42),
            num_ctx: Some(4096),
            num_predict: Some(2048),
            ..Default::default()
        },
    ]
}
