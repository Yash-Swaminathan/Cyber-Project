import React from 'react';

function StatisticCard({ title, value, icon, trend, trendValue, color = 'blue' }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
    yellow: 'bg-yellow-500',
    purple: 'bg-purple-500',
  };
  
  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→';
  const trendColorClass = trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : 'text-gray-500';
  
  return (
    <div className="bg-white rounded-lg shadow p-4 flex flex-col">
      <div className="flex items-center mb-2">
        <div className={`${colorClasses[color]} p-2 rounded text-white mr-3`}>
          {icon}
        </div>
        <h3 className="text-gray-500 text-sm font-medium">{title}</h3>
      </div>
      
      <div className="flex justify-between items-end mt-2">
        <div className="text-2xl font-bold">{value}</div>
        {trend && (
          <div className={`flex items-center ${trendColorClass}`}>
            <span className="text-lg mr-1">{trendIcon}</span>
            <span>{trendValue}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default StatisticCard;