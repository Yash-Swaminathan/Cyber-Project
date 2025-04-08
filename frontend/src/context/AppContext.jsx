import React, { createContext, useState, useEffect, useCallback } from 'react';
import { 
  fetchConfig, 
  fetchHealthStatus, 
  processFlows, 
  sendTestAlert 
} from '../utils/api';
import useWebSocket from '../hooks/useWebSocket';

export const AppContext = createContext();

export const AppProvider = ({ children }) => {
  // Global state
  const [config, setConfig] = useState(null);
  const [healthStatus, setHealthStatus] = useState({ status: 'unknown' });
  const [alerts, setAlerts] = useState([]);
  const [recentFlows, setRecentFlows] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [statistics, setStatistics] = useState({
    totalFlows: 0,
    totalAlerts: 0,
    highSeverityCount: 0,
    mediumSeverityCount: 0,
    lowSeverityCount: 0
  });

  // WebSocket connection for real-time updates
  const { lastMessage, connectionStatus } = useWebSocket('ws://localhost:8000/ws');

  // Load initial configuration and health status
  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true);
      try {
        const [configData, healthData] = await Promise.all([
          fetchConfig(),
          fetchHealthStatus()
        ]);
        setConfig(configData);
        setHealthStatus(healthData);
      } catch (err) {
        setError(`Failed to initialize: ${err.message}`);
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    initialize();
  }, []);

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage);
        
        if (data.type === 'alert') {
          // Add new alert to state
          setAlerts(prev => [data.alert, ...prev]);
          
          // Update statistics
          setStatistics(prev => ({
            ...prev,
            totalAlerts: prev.totalAlerts + 1,
            [`${data.alert.severity}SeverityCount`]: 
              prev[`${data.alert.severity}SeverityCount`] + 1
          }));
        } 
        else if (data.type === 'flow') {
          // Add new flow data
          setRecentFlows(prev => {
            const newFlows = [...data.flows, ...prev].slice(0, 100); // Keep last 100 flows
            return newFlows;
          });
          
          // Update flow statistics
          setStatistics(prev => ({
            ...prev,
            totalFlows: prev.totalFlows + data.flows.length
          }));
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    }
  }, [lastMessage]);

  // Function to submit flows for detection
  const submitFlows = useCallback(async (flowsData) => {
    setIsLoading(true);
    try {
      const response = await processFlows(flowsData);
      
      if (response.alerts && response.alerts.length > 0) {
        setAlerts(prev => [...response.alerts, ...prev]);
      }
      
      return response;
    } catch (err) {
      setError(`Error processing flows: ${err.message}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Function to send test alert
  const triggerTestAlert = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await sendTestAlert();
      return response;
    } catch (err) {
      setError(`Error sending test alert: ${err.message}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Function to refresh health status
  const checkHealth = useCallback(async () => {
    try {
      const healthData = await fetchHealthStatus();
      setHealthStatus(healthData);
      return healthData;
    } catch (err) {
      setError(`Error checking health: ${err.message}`);
      throw err;
    }
  }, []);

  // Clear a specific alert
  const clearAlert = useCallback((alertId) => {
    setAlerts(prev => prev.filter(alert => alert.alert_id !== alertId));
  }, []);

  // Refresh configuration
  const refreshConfig = useCallback(async () => {
    try {
      const configData = await fetchConfig();
      setConfig(configData);
      return configData;
    } catch (err) {
      setError(`Error refreshing config: ${err.message}`);
      throw err;
    }
  }, []);

  // Context value
  const contextValue = {
    config,
    healthStatus,
    alerts,
    recentFlows,
    statistics,
    isLoading,
    error,
    connectionStatus,
    submitFlows,
    triggerTestAlert,
    checkHealth,
    clearAlert,
    refreshConfig
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};