import React, { useContext, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AppContext } from '../../context/AppContext';

const AlertDetails = () => {
  const { alertId } = useParams();
  const navigate = useNavigate();
  const { alerts, clearAlert } = useContext(AppContext);

  // Find the specific alert
  const alert = useMemo(() => {
    return alerts.find(a => a.alert_id === alertId);
  }, [alerts, alertId]);

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Handle dismiss alert
  const handleDismiss = () => {
    clearAlert(alertId);
    navigate('/alerts');
  };

  if (!alert) {
    return (
      <div className="p-6">
        <div className="bg-white p-8 rounded-lg shadow-md text-center">
          <h2 className="text-xl font-medium text-gray-700 mb-4">Alert Not Found</h2>
          <p className="text-gray-500 mb-4">The alert you're looking for doesn't exist or has been dismissed</p>
          <button
            onClick={() => navigate('/alerts')}
            className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
          >
            Back to Alerts
          </button>
        </div>
      </div>
    );
  }

  // Get severity color
  const getSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'bg-red-600';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <button
          onClick={() => navigate('/alerts')}
          className="flex items-center text-indigo-600 hover:text-indigo-800"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Alerts
        </button>
      </div>

      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        {/* Alert Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex justify-between items-start">
            <div>
              <div className="flex items-center mb-2">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white ${getSeverityColor(alert.severity)}`}>
                  {alert.severity.toUpperCase()} SEVERITY
                </span>
                <span className="ml-3 text-sm text-gray-500">
                  Alert ID: {alert.alert_id}
                </span>
              </div>
              <h2 className="text-xl font-medium text-gray-800">{alert.description}</h2>
            </div>
            <button
              onClick={handleDismiss}
              className="px-3 py-1 bg-red-50 text-red-600 rounded-md hover:bg-red-100"
            >
              Dismiss Alert
            </button>
          </div>
        </div>

        {/* Alert Details */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-medium text-gray-800 mb-4">Alert Information</h3>
            <table className="min-w-full">
              <tbody>
                <tr>
                  <td className="py-2 pr-4 text-sm font-medium text-gray-500">Timestamp</td>
                  <td className="py-2 text-sm text-gray-900">{formatTimestamp(alert.timestamp)}</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 text-sm font-medium text-gray-500">Anomaly Score</td>
                  <td className="py-2 text-sm text-gray-900">{alert.anomaly_score.toFixed(4)}</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 text-sm font-medium text-gray-500">Affected Flows</td>
                  <td className="py-2 text-sm text-gray-900">{alert.affected_flows?.length || 0}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div>
            <h3 className="text-lg font-medium text-gray-800 mb-4">Recommended Actions</h3>
            <ul className="text-sm text-gray-600 space-y-2">
              <li className="flex items-start">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                Investigate suspicious traffic patterns between the affected IPs
              </li>
              <li className="flex items-start">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                Review logs for the affected systems
              </li>
              <li className="flex items-start">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                Check for known vulnerabilities in services running on the affected ports
              </li>
            </ul>
          </div>
        </div>

        {/* Affected Network Flows */}
        <div className="p-6 border-t border-gray-200">
          <h3 className="text-lg font-medium text-gray-800 mb-4">Affected Network Flows</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Source IP
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Destination IP
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Protocol
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Source Port
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Destination Port
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Bytes Sent/Received
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {alert.affected_flows?.map((flow, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {flow.src_ip}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {flow.dst_ip}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {flow.protocol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {flow.src_port}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {flow.dst_port}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {flow.bytes_sent} / {flow.bytes_received}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertDetails;