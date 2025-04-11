import React, { createContext, useState, useEffect, useCallback } from 'react';
import { fetchConfig, updateConfigOnServer, sendTestAlert as sendTestAlertApi } from '../utils/api';

// Dummy fetchAlerts for testing purposes.
const fetchAlerts = async () => {
  return [
    {
      alert_id: 'test-alert',
      timestamp: new Date().toISOString(),
      anomaly_score: 0.5,
      severity: 'medium',
      description: 'This is a test alert'
    }
  ];
};

export const AppContext = createContext({});

export const AppProvider = ({ children }) => {
  // Config state for the Config panel.
  const [config, setConfig] = useState(null);

  // Other states
  const [alerts, setAlerts] = useState([]);
  const [healthStatus, setHealthStatus] = useState({});
  const [statistics, setStatistics] = useState({});
  const [recentFlows, setRecentFlows] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // getConfig wrapped in useCallback to keep it stable.
  const getConfig = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await fetchConfig();
      setConfig(data);
      return data;
    } catch (err) {
      console.error("Error loading configuration:", err);
      setError("Failed to load configuration");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load configuration once on mount.
  useEffect(() => {
    getConfig();
  }, [getConfig]);

  // Load alerts on mount using dummy fetchAlerts.
  useEffect(() => {
    async function loadAlerts() {
      setIsLoading(true);
      try {
        const data = await fetchAlerts();
        setAlerts(
          data && data.length > 0
            ? data
            : [{
                alert_id: 'test-alert',
                timestamp: new Date().toISOString(),
                anomaly_score: 0.5,
                severity: 'medium',
                description: 'This is a test alert'
              }]
        );
      } catch (err) {
        console.error('Error fetching alerts in AppContext:', err);
        setAlerts([{
          alert_id: 'test-alert',
          timestamp: new Date().toISOString(),
          anomaly_score: 0.5,
          severity: 'medium',
          description: 'This is a test alert'
        }]);
      } finally {
        setIsLoading(false);
      }
    }
    loadAlerts();
  }, []);

  // Update dummy health status and statistics based on alerts.
  useEffect(() => {
    setHealthStatus({ status: 'healthy', model_loaded: true });
    setStatistics({
      totalFlows: 1000,
      totalAlerts: alerts.length,
      highSeverityCount: alerts.filter(a => a.severity.toLowerCase() === 'high').length
    });
  }, [alerts]);

  // Load recent network flows (dummy data).
  useEffect(() => {
    async function loadFlows() {
      try {
        // Replace with actual API call if available.
        setRecentFlows([
          {
            src_ip: '192.168.0.10',
            dst_ip: '10.0.0.5',
            protocol: 'TCP',
            bytes_sent: 1000,
            bytes_received: 2000
          }
        ]);
      } catch (err) {
        console.error('Error loading flows', err);
      }
    }
    loadFlows();
  }, []);

  // Function to update configuration on the server.
  const updateConfig = async (newConfig) => {
    setIsLoading(true);
    try {
      const updated = await updateConfigOnServer(newConfig);
      setConfig(updated);
      return updated;
    } catch (err) {
      console.error("Failed to update configuration:", err);
      setError("Failed to update configuration");
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  // Function to send a test alert.
  const sendTestAlert = async (alertData) => {
    setIsLoading(true);
    try {
      const response = await sendTestAlertApi(alertData);
      return response;
    } catch (err) {
      console.error("Error sending test alert:", err);
      setError("Error sending test alert");
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    config,
    getConfig,
    updateConfig,
    alerts,
    healthStatus,
    statistics,
    recentFlows,
    isLoading,
    error,
    setAlerts,
    sendTestAlert
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};
