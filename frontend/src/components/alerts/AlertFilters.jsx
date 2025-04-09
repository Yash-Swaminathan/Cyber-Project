import React, { useState } from 'react';
import { Filter } from 'lucide-react';

function AlertFilters({ onFilterChange }) {
  const [timeRange, setTimeRange] = useState('24h');
  const [severity, setSeverity] = useState('all');
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  
  const handleApplyFilters = () => {
    onFilterChange({ timeRange, severity });
    if (window.innerWidth < 768) {
      setIsFilterOpen(false);
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow mb-4">
      <div className="p-4 border-b flex justify-between items-center">
        <h3 className="font-medium">Filters</h3>
        <button 
          className="md:hidden bg-gray-100 p-2 rounded-full"
          onClick={() => setIsFilterOpen(!isFilterOpen)}
        >
          <Filter size={16} />
        </button>
      </div>
      
      <div className={`p-4 space-y-4 ${isFilterOpen || window.innerWidth >= 768 ? 'block' : 'hidden'} md:block`}>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Time Range</label>
          <select 
            className="w-full border border-gray-300 rounded-md p-2"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Severity</label>
          <select 
            className="w-full border border-gray-300 rounded-md p-2"
            value={severity}
            onChange={(e) => setSeverity(e.target.value)}
          >
            <option value="all">All Severities</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>
        
        <button 
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
          onClick={handleApplyFilters}
        >
          Apply Filters
        </button>
      </div>
    </div>
  );
}

export default AlertFilters;