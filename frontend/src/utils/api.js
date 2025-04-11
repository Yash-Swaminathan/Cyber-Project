// API base URL configuration
const API_BASE_URL = '/api/v1';

// Helper function for API calls
const apiCall = async (endpoint, method = 'GET', data = null) => {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };

  if (data) {
    options.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
};

// API functions matching backend endpoints
export const fetchHealthStatus = () => apiCall('/health');
export const fetchConfig = () => apiCall('/config');
export const processFlows = (flows) => apiCall('/detect', 'POST', { flows });
export const sendTestAlert = (alertData) => apiCall('/test-alert', 'POST', alertData);

// NEW: Update configuration on the server (expects a PUT endpoint)
export const updateConfigOnServer = (newConfig) => apiCall('/config', 'PUT', newConfig);

// Function to format network flow data for submission
export const formatFlowData = (rawFlowData) => {
  return {
    timestamp: rawFlowData.timestamp || new Date().toISOString(),
    src_ip: rawFlowData.src_ip,
    dst_ip: rawFlowData.dst_ip,
    src_port: parseInt(rawFlowData.src_port),
    dst_port: parseInt(rawFlowData.dst_port),
    protocol: rawFlowData.protocol,
    bytes_sent: parseInt(rawFlowData.bytes_sent),
    bytes_received: parseInt(rawFlowData.bytes_received),
    packets_sent: parseInt(rawFlowData.packets_sent),
    packets_received: parseInt(rawFlowData.packets_received),
    duration: parseFloat(rawFlowData.duration),
    additional_features: rawFlowData.additional_features || null,
  };
};

// Batch submission utility
export const submitFlowBatch = (flowBatch) => {
  const formattedFlows = flowBatch.map(formatFlowData);
  return processFlows(formattedFlows);
};
