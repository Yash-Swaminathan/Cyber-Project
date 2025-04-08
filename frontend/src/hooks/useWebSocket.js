import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (url) => {
  const [lastMessage, setLastMessage] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }

    try {
      ws.current = new WebSocket(url);
      setConnectionStatus('connecting');

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        
        // Clear any pending reconnect timeouts
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.current.onmessage = (event) => {
        setLastMessage(event.data);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };

      ws.current.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect after a delay
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...');
          connect();
        }, 5000); // 5 second reconnection delay
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
      setConnectionStatus('error');
      
      // Attempt to reconnect after a delay
      reconnectTimeoutRef.current = setTimeout(connect, 5000);
    }
  }, [url]);

  // Send message through the WebSocket
  const sendMessage = useCallback((data) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(typeof data === 'string' ? data : JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  // Connect WebSocket on component mount
  useEffect(() => {
    connect();

    // Clean up WebSocket on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, [connect]);

  return {
    lastMessage,
    connectionStatus,
    sendMessage
  };
};

export default useWebSocket;