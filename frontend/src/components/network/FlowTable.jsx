import React from 'react';

function FlowTable({ flows, onSelectFlow }) {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="p-4 border-b">
        <h2 className="text-lg font-medium">Network Flows</h2>
      </div>
      
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
                Packets
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Bytes
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Anomaly Score
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {flows.map((flow) => (
              <tr 
                key={flow.id} 
                className="hover:bg-gray-50 cursor-pointer"
                onClick={() => onSelectFlow(flow)}
              >
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {flow.source_ip}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {flow.destination_ip}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {flow.protocol}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {flow.packets}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {flow.bytes}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <span 
                      className={`inline-block h-2 w-2 rounded-full mr-2 ${
                        flow.anomaly_score > 0.8 ? 'bg-red-500' : 
                        flow.anomaly_score > 0.5 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                    ></span>
                    <span className="text-sm text-gray-900">{flow.anomaly_score.toFixed(2)}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default FlowTable;