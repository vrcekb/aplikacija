//! WebSocket Message Handlers
//!
//! Event-driven message handling for WebSocket connections.

use crate::error::NetworkResult;
use crate::types::{ConnectionId, WebSocketMessage, WebSocketCloseFrame};
use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Type alias for async message handler functions
pub type MessageHandlerFn = Arc<dyn Fn(ConnectionId, WebSocketMessage) -> Pin<Box<dyn Future<Output = NetworkResult<()>> + Send>> + Send + Sync>;

/// Type alias for async error handler functions
pub type ErrorHandlerFn = Arc<dyn Fn(ConnectionId, crate::error::NetworkError) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Type alias for async close handler functions
pub type CloseHandlerFn = Arc<dyn Fn(ConnectionId, Option<WebSocketCloseFrame>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Type alias for async connect handler functions
pub type ConnectHandlerFn = Arc<dyn Fn(ConnectionId) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// WebSocket message handler trait
#[async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle incoming message
    ///
    /// # Errors
    /// Returns error if message handling fails
    async fn handle_message(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()>;

    /// Handle connection error
    async fn handle_error(&self, connection_id: ConnectionId, error: crate::error::NetworkError);

    /// Handle connection close
    async fn handle_close(&self, connection_id: ConnectionId, close_frame: Option<WebSocketCloseFrame>);

    /// Handle successful connection
    async fn handle_connect(&self, connection_id: ConnectionId);
}

/// WebSocket event handlers collection
#[derive(Clone)]
pub struct WebSocketHandlers {
    /// Message handler
    pub on_message: Option<MessageHandlerFn>,
    /// Error handler
    pub on_error: Option<ErrorHandlerFn>,
    /// Close handler
    pub on_close: Option<CloseHandlerFn>,
    /// Connect handler
    pub on_connect: Option<ConnectHandlerFn>,
}

impl WebSocketHandlers {
    /// Create new empty handlers
    #[must_use]
    pub const fn new() -> Self {
        Self {
            on_message: None,
            on_error: None,
            on_close: None,
            on_connect: None,
        }
    }

    /// Set message handler
    #[must_use]
    pub fn message<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(ConnectionId, WebSocketMessage) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NetworkResult<()>> + Send + 'static,
    {
        self.on_message = Some(Arc::new(move |conn_id, msg| {
            Box::pin(handler(conn_id, msg))
        }));
        self
    }

    /// Set error handler
    #[must_use]
    pub fn error<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(ConnectionId, crate::error::NetworkError) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_error = Some(Arc::new(move |conn_id, err| {
            Box::pin(handler(conn_id, err))
        }));
        self
    }

    /// Set close handler
    #[must_use]
    pub fn close<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(ConnectionId, Option<WebSocketCloseFrame>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_close = Some(Arc::new(move |conn_id, frame| {
            Box::pin(handler(conn_id, frame))
        }));
        self
    }

    /// Set connect handler
    #[must_use]
    pub fn connect<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(ConnectionId) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.on_connect = Some(Arc::new(move |conn_id| {
            Box::pin(handler(conn_id))
        }));
        self
    }
}

impl Default for WebSocketHandlers {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MessageHandler for WebSocketHandlers {
    async fn handle_message(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()> {
        if let Some(handler) = &self.on_message {
            handler(connection_id, message).await
        } else {
            Ok(())
        }
    }

    async fn handle_error(&self, connection_id: ConnectionId, error: crate::error::NetworkError) {
        if let Some(handler) = &self.on_error {
            handler(connection_id, error).await;
        }
    }

    async fn handle_close(&self, connection_id: ConnectionId, close_frame: Option<WebSocketCloseFrame>) {
        if let Some(handler) = &self.on_close {
            handler(connection_id, close_frame).await;
        }
    }

    async fn handle_connect(&self, connection_id: ConnectionId) {
        if let Some(handler) = &self.on_connect {
            handler(connection_id).await;
        }
    }
}

/// Default message handler implementation
pub struct DefaultMessageHandler;

#[async_trait]
impl MessageHandler for DefaultMessageHandler {
    async fn handle_message(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()> {
        tracing::debug!("Received message on connection {}: {:?}", connection_id, message);
        
        match message {
            WebSocketMessage::Text(text) => {
                tracing::info!("Text message: {}", text);
            }
            WebSocketMessage::Binary(data) => {
                tracing::info!("Binary message: {} bytes", data.len());
            }
            WebSocketMessage::Ping(data) => {
                tracing::debug!("Ping received: {} bytes", data.len());
                // TODO: Automatically send pong response
            }
            WebSocketMessage::Pong(data) => {
                tracing::debug!("Pong received: {} bytes", data.len());
            }
            WebSocketMessage::Close(close_frame) => {
                if let Some(frame) = close_frame {
                    tracing::info!("Close frame: code={}, reason={}", frame.code, frame.reason);
                } else {
                    tracing::info!("Close frame without details");
                }
            }
        }
        
        Ok(())
    }

    async fn handle_error(&self, connection_id: ConnectionId, error: crate::error::NetworkError) {
        tracing::error!("WebSocket error on connection {}: {}", connection_id, error);
    }

    async fn handle_close(&self, connection_id: ConnectionId, close_frame: Option<WebSocketCloseFrame>) {
        if let Some(frame) = close_frame {
            tracing::info!("Connection {} closed: code={}, reason={}", connection_id, frame.code, frame.reason);
        } else {
            tracing::info!("Connection {} closed without close frame", connection_id);
        }
    }

    async fn handle_connect(&self, connection_id: ConnectionId) {
        tracing::info!("WebSocket connection established: {}", connection_id);
    }
}

/// Echo message handler for testing
pub struct EchoMessageHandler;

#[async_trait]
impl MessageHandler for EchoMessageHandler {
    async fn handle_message(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()> {
        tracing::debug!("Echo handler received message on connection {}", connection_id);
        
        match message {
            WebSocketMessage::Text(text) => {
                tracing::info!("Echoing text: {}", text);
                // TODO: Send echo response back
            }
            WebSocketMessage::Binary(data) => {
                tracing::info!("Echoing binary: {} bytes", data.len());
                // TODO: Send echo response back
            }
            WebSocketMessage::Ping(_data) => {
                tracing::debug!("Responding to ping with pong");
                // TODO: Send pong response
            }
            _ => {
                // Don't echo pong, close, etc.
            }
        }
        
        Ok(())
    }

    async fn handle_error(&self, connection_id: ConnectionId, error: crate::error::NetworkError) {
        tracing::error!("Echo handler error on connection {}: {}", connection_id, error);
    }

    async fn handle_close(&self, connection_id: ConnectionId, close_frame: Option<WebSocketCloseFrame>) {
        tracing::info!("Echo handler connection {} closed", connection_id);
        if let Some(frame) = close_frame {
            tracing::debug!("Close details: code={}, reason={}", frame.code, frame.reason);
        }
    }

    async fn handle_connect(&self, connection_id: ConnectionId) {
        tracing::info!("Echo handler connection established: {}", connection_id);
    }
}

/// JSON message handler for structured data
pub struct JsonMessageHandler<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> JsonMessageHandler<T>
where
    T: serde::de::DeserializeOwned + Send + Sync + 'static,
{
    /// Create new JSON message handler
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Default for JsonMessageHandler<T>
where
    T: serde::de::DeserializeOwned + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T> MessageHandler for JsonMessageHandler<T>
where
    T: serde::de::DeserializeOwned + Send + Sync + 'static,
{
    async fn handle_message(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()> {
        match message {
            WebSocketMessage::Text(text) => {
                match serde_json::from_str::<T>(&text) {
                    Ok(_parsed) => {
                        tracing::debug!("Successfully parsed JSON message on connection {}", connection_id);
                        // TODO: Handle parsed message
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse JSON message: {}", e);
                        return Err(crate::error::NetworkError::Serialization {
                            format: "json".to_string(),
                            message: e.to_string(),
                        });
                    }
                }
            }
            WebSocketMessage::Binary(data) => {
                match serde_json::from_slice::<T>(&data) {
                    Ok(_parsed) => {
                        tracing::debug!("Successfully parsed JSON from binary message on connection {}", connection_id);
                        // TODO: Handle parsed message
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse JSON from binary: {}", e);
                        return Err(crate::error::NetworkError::Serialization {
                            format: "json".to_string(),
                            message: e.to_string(),
                        });
                    }
                }
            }
            _ => {
                // Handle other message types with default behavior
            }
        }
        
        Ok(())
    }

    async fn handle_error(&self, connection_id: ConnectionId, error: crate::error::NetworkError) {
        tracing::error!("JSON handler error on connection {}: {}", connection_id, error);
    }

    async fn handle_close(&self, connection_id: ConnectionId, close_frame: Option<WebSocketCloseFrame>) {
        tracing::info!("JSON handler connection {} closed", connection_id);
        if let Some(frame) = close_frame {
            tracing::debug!("Close details: code={}, reason={}", frame.code, frame.reason);
        }
    }

    async fn handle_connect(&self, connection_id: ConnectionId) {
        tracing::info!("JSON handler connection established: {}", connection_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WebSocketMessage;

    #[tokio::test]
    async fn test_default_handler() {
        let handler = DefaultMessageHandler;
        let connection_id = ConnectionId::new();
        let message = WebSocketMessage::Text("Hello".to_string());

        let result = handler.handle_message(connection_id, message).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_echo_handler() {
        let handler = EchoMessageHandler;
        let connection_id = ConnectionId::new();
        let message = WebSocketMessage::Text("Echo test".to_string());

        let result = handler.handle_message(connection_id, message).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_json_handler() {
        #[derive(serde::Deserialize)]
        struct TestMessage {
            #[allow(dead_code)]
            content: String,
        }

        let handler = JsonMessageHandler::<TestMessage>::new();
        let connection_id = ConnectionId::new();
        
        // Valid JSON
        let valid_json = WebSocketMessage::Text(r#"{"content": "test"}"#.to_string());
        let result = handler.handle_message(connection_id, valid_json).await;
        assert!(result.is_ok());

        // Invalid JSON
        let invalid_json = WebSocketMessage::Text("invalid json".to_string());
        let result = handler.handle_message(connection_id, invalid_json).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_websocket_handlers_builder() {
        let handlers = WebSocketHandlers::new()
            .message(|_conn_id, _msg| async { Ok(()) })
            .error(|_conn_id, _err| async {})
            .close(|_conn_id, _frame| async {})
            .connect(|_conn_id| async {});

        assert!(handlers.on_message.is_some());
        assert!(handlers.on_error.is_some());
        assert!(handlers.on_close.is_some());
        assert!(handlers.on_connect.is_some());
    }

    #[tokio::test]
    async fn test_handlers_execution() {
        let connection_id = ConnectionId::new();
        let message = WebSocketMessage::Text("test".to_string());
        let error = crate::error::NetworkError::internal("test error");

        let handlers = WebSocketHandlers::new()
            .message(|_conn_id, _msg| async { Ok(()) })
            .error(|_conn_id, _err| async {})
            .close(|_conn_id, _frame| async {})
            .connect(|_conn_id| async {});

        // Test message handling
        let result = handlers.handle_message(connection_id, message).await;
        assert!(result.is_ok());

        // Test error handling
        handlers.handle_error(connection_id, error).await;

        // Test close handling
        handlers.handle_close(connection_id, None).await;

        // Test connect handling
        handlers.handle_connect(connection_id).await;
    }
}
