import React, { useContext, useState } from 'react';
import { AppContext } from '../../context/AppContext';

function TestAlert() {
  const { sendTestAlert } = useContext(AppContext);
  const [severity, setSeverity] = useState('medium');
  const [message, setMessage] = useState('This is a test alert');
  const [isSending, setIsSending] = useState(false);
  const [result, setResult] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSending(true);
    setResult(null);
    
    try {
      const response = await sendTestAlert({ severity, message });
      setResult({ success: true, data: response });
    } catch (error) {
      setResult({ success: false, error: error.message });
    } finally {
      setIsSending(false);
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="p-4 border-b">
        <h2 className="text-lg font-medium">Test Alert</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Alert Severity
          </label>
          <select
            value={severity}
            onChange={(e) => setSeverity(e.target.value)}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Alert Message
          </label>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={3}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
          />
        </div>
        
        <div className="flex items-center justify-between pt-2">
          <button
            type="submit"
            disabled={isSending}
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {isSending ? 'Sending...' : 'Send Test Alert'}
          </button>
        </div>
      </form>
      
      {result && (
        <div className="px-4 pb-4">
          <div className={`p-3 rounded-md ${result.success ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
            {result.success ? (
              <p>Test alert sent successfully!</p>
            ) : (
              <p>Error sending test alert: {result.error}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default TestAlert;