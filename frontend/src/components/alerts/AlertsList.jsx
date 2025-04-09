import React, { useContext, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { AppContext } from '../../context/AppContext';

const AlertsList = () => {
  const { alerts, clearAlert } = useContext(AppContext);
  const [filters, setFilters] = useState({
    severity: 'all',
    timeRange: 'all',
    searchTerm: ''
  });

  // Format timestamp for display
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Get severity badge style
  const getSeverityBadge = (severity) => {
    const baseClasses = 'px-2 py-1 text-xs font-medium rounded-full text-white';
    switch (severity.toLowerCase()) {
      case 'high':
        return `${baseClasses} bg-red-600`;
      case 'medium':
        return `${baseClasses} bg-yellow-500`;
      case 'low':
        return `${baseClasses} bg-blue-500`;
      default:
        return `${baseClasses} bg-gray-500`;
    }
  };

  // Filter alerts based on current filters
  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      // Filter by severity
      if (filters.severity !== 'all' && alert.severity !== filters.severity) {
        return false;
      }

      // Filter by time range
      if (filters.timeRange !== 'all') {
        const alertTime = new Date(alert.timestamp).getTime();
        const now = new Date().getTime();
        const hourInMs = 60 * 60 * 1000;
        const dayInMs = 24 * hourInMs;

        switch (filters.timeRange) {
          case '1h':
            if (now - alertTime > hourInMs) return false;
            break;
          case '24h':
            if (now - alertTime > dayInMs) return false;
            break;
          case '7d':
            if (now - alertTime > 7 * dayInMs) return false;
            break;
          default:
            break;
        }
      }

      // Filter by search term (check IP addresses, descriptions)
      if (filters.searchTerm) {
        const searchLower = filters.searchTerm.toLowerCase();
        const descriptionMatch = alert.description.toLowerCase().includes(searchLower);
        const ipMatch = alert.affected_flows.some(flow => 
          flow.src_ip?.toLowerCase().includes(searchLower) || 
          flow.dst_ip?.toLowerCase().includes(searchLower)
        );
        
        if (!descriptionMatch && !ipMatch) {
          return false;
        }
      }

      return true;
    });
  }, [alerts, filters]);

  // Handle filter changes
  const handleFilterChange = (filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value
    }));
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold">Network Anomaly Alerts</h2>
        <div className="text-sm text-gray-600">
          Showing {filteredAlerts.length} of {alerts.length} alerts
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow-md mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label htmlFor="severity" className="block text-sm font-medium text-gray-700 mb-1">
              Severity
            </label>
            <select
              id="severity"
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              value={filters.severity}
              onChange={(e) => handleFilterChange('severity', e.target.value)}
            >
              <option value="all">All Severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>

          <div>
            <label htmlFor="timeRange" className="block text-sm font-medium text-gray-700 mb-1">
              Time Range
            </label>
            <select
              id="timeRange"
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              value={filters.timeRange}
              onChange={(e) => handleFilterChange('timeRange', e.target.value)}
            >
              <option value="all">All Time</option>
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
            </select>
          </div>

          <div className="flex-grow">
            <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-1">
              Search
            </label>
            <input
              type="text"
              id="search"
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Search by IP or description..."
              value={filters.searchTerm}
              onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Alerts List */}
      {filteredAlerts.length > 0 ? (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Severity
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Anomaly Score
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Description
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Affected IPs
                </th>
                <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAlerts.map((alert) => (
                <tr key={alert.alert_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={getSeverityBadge(alert.severity)}>
                      {alert.severity}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatTimestamp(alert.timestamp)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {alert.anomaly_score.toFixed(4)}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
                    {alert.description}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {alert.affected_flows && alert.affected_flows.length > 0 ? (
                      <div className="flex flex-col space-y-1">
                        {alert.affected_flows.slice(0, 2).map((flow, idx) => (
                          <div key={idx} className="text-xs">
                            {flow.src_ip} → {flow.dst_ip}
                          </div>
                        ))}
                        {alert.affected_flows.length > 2 && (
                          <div className="text-xs text-gray-400">
                            +{alert.affected_flows.length - 2} more
                          </div>
                        )}
                      </div>
                    ) : (
                      <span>-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <Link 
                      to={`/alerts/${alert.alert_id}`}
                      className="text-indigo-600 hover:text-indigo-900 mr-4"
                    >
                      Details
                    </Link>
                    <button
                      onClick={() => clearAlert(alert.alert_id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      Dismiss
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-white p-8 rounded-lg shadow-md text-center">
          <div className="text-gray-500 mb-2">No alerts found matching your filters</div>
          {alerts.length > 0 && filters.severity !== 'all' && (
            <button
              onClick={() => setFilters(prev => ({ ...prev, severity: 'all' }))}
              className="text-indigo-600 hover:text-indigo-800"
            >
              Clear severity filter
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default AlertsList;