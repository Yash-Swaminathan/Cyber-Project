import React, { createContext, useState, useCallback } from 'react';
import {
  fetchConfig,
  fetchHealthStatus,
  processFlows,
  sendTestAlert,
  submitFlowBatch
} from '../utils/api';


export const APIContext = createContext({});

export const APIProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const fetchNetworkFlows = useCallback(async (params = {}) => {
    try {
      setIsLoading(true);
      const response = await api.get('/api/v1/flows', { params });
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to fetch network flows');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);  
  
  const fetchAlerts = useCallback(async (params = {}) => {
    try {
      setIsLoading(true);
      const response = await api.get('/api/v1/alerts', { params });
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to fetch alerts');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  const getAlertDetails = useCallback(async (alertId) => {
    try {
      setIsLoading(true);
      const response = await api.get(`/api/v1/alerts/${alertId}`);
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to fetch alert details');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  const getSystemHealth = useCallback(async () => {
    try {
      const response = await api.get('/api/v1/health');
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to fetch system health');
      throw err;
    }
  }, []);
  
  const getSystemMetrics = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await api.get('/api/v1/metrics');
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to fetch system metrics');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  const detectAnomaly = useCallback(async (flowData) => {
    try {
      setIsLoading(true);
      const response = await api.post('/api/v1/detect', flowData);
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to run anomaly detection');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  const value = {
    isLoading,
    error,
    fetchNetworkFlows,
    fetchAlerts,
    getAlertDetails,
    getSystemHealth,
    getSystemMetrics,
    detectAnomaly,
    clearError: () => setError(null)
  };
  
  return (
    <APIContext.Provider value={value}>
      {children}
    </APIContext.Provider>
  );
};