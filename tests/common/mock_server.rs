// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Minimal raw-TCP HTTP mock: the tests drive byte-level chunking of
//! the response body, which no mock crate can do.

use std::sync::Arc;
use std::sync::Mutex;

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// Same as `mock_server_capture`, but the captured request is dropped.
pub async fn mock_server(
    status: &'static str,
    content_type: &'static str,
    chunks: Vec<Vec<u8>>,
) -> (String, JoinHandle<()>) {
    let (host, handle, _) = mock_server_capture(status, content_type, chunks).await;
    (host, handle)
}

/// Serves one request and answers with the given status and body
/// chunks (each chunk goes out as a separate write, 1 ms apart —
/// that is what drives the chunk-split tests). Captures the full
/// request (headers and body) for the caller to assert on.
pub async fn mock_server_capture(
    status: &'static str,
    content_type: &'static str,
    chunks: Vec<Vec<u8>>,
) -> (String, JoinHandle<()>, Arc<Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let captured: Arc<Mutex<String>> = Arc::default();
    let captured_handle = captured.clone();

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 1024];

        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0, "connection closed before request headers");
            request.extend_from_slice(&buffer[..read]);
        }

        // The body may already sit in the header buffer — read the
        // rest by Content-Length before reporting the request.
        let header_end = request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let headers_text = String::from_utf8_lossy(&request[..header_end]).to_string();
        let content_length: usize = headers_text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().ok())
            })
            .flatten()
            .unwrap_or(0);
        while request.len() < header_end + content_length {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0, "connection closed before request body");
            request.extend_from_slice(&buffer[..read]);
        }
        *captured_handle.lock().unwrap() = String::from_utf8_lossy(&request).to_string();

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

    (format!("http://{}", address), handle, captured)
}
