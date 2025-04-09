import React from 'react';
import { X } from 'lucide-react';

function FlowDetails({ flow, onClose }) {
  if (!flow) return null;
  
  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="text-lg font-medium">Flow Details</h2>
        <button 
          onClick={onClose} 
          className="text-gray-500 hover:text-gray-700"
        >
          <X size={20} />
        </button>
      </div>
      
      <div className="p-4 space-y-4">
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">ID:</span>
          <span className="font-medium">{flow.id}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Source IP:</span>
          <span className="font-medium">{flow.source_ip}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Destination IP:</span>
          <span className="font-medium">{flow.destination_ip}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Protocol:</span>
          <span className="font-medium">{flow.protocol}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Source Port:</span>
          <span className="font-medium">{flow.source_port}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Destination Port:</span>
          <span className="font-medium">{flow.destination_port}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Packets:</span>
          <span className="font-medium">{flow.packets}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Bytes:</span>
          <span className="font-medium">{flow.bytes}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Start Time:</span>
          <span className="font-medium">{new Date(flow.start_time).toLocaleString()}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">End Time:</span>
          <span className="font-medium">{new Date(flow.end_time).toLocaleString()}</span>
        </div>
        
        <div className="flex justify-between border-b pb-2">
          <span className="text-gray-600">Anomaly Score:</span>
          <span className={`font-medium ${
            flow.anomaly_score > 0.8 ? 'text-red-600' : 
            flow.anomaly_score > 0.5 ? 'text-yellow-600' : 'text-green-600'
          }`}>
            {flow.anomaly_score.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

export default FlowDetails;