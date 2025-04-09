import React, { useState } from 'react';

function AlertConfig({ config, onSave }) {
  const [thresholds, setThresholds] = useState({
    low: config.thresholds.low,
    medium: config.thresholds.medium,
    high: config.thresholds.high
  });
  
  const [notificationSettings, setNotificationSettings] = useState({
    email: config.notifications.email,
    slack: config.notifications.slack,
    webhook: config.notifications.webhook
  });
  
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  const handleThresholdChange = (level, value) => {
    setThresholds(prev => ({
      ...prev,
      [level]: parseFloat(value)
    }));
  };
  
  const handleNotificationToggle = (type) => {
    setNotificationSettings(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    
    const newConfig = {
      thresholds,
      notifications: notificationSettings
    };
    
    const success = await onSave(newConfig);
    
    setIsSaving(false);
    if (success) {
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="p-4 border-b">
        <h2 className="text-lg font-medium">Alert Configuration</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 space-y-6">
        <div>
          <h3 className="text-md font-medium mb-3">Anomaly Score Thresholds</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Low Severity (0.0 - 1.0)
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={thresholds.low}
                onChange={(e) => handleThresholdChange('low', e.target.value)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Medium Severity (0.0 - 1.0)
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={thresholds.medium}
                onChange={(e) => handleThresholdChange('medium', e.target.value)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">
                High Severity (0.0 - 1.0)
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={thresholds.high}
                onChange={(e) => handleThresholdChange('high', e.target.value)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
              />
            </div>
          </div>
        </div>
        
        <div>
          <h3 className="text-md font-medium mb-3">Notification Channels</h3>
          
          <div className="space-y-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="email-notifications"
                checked={notificationSettings.email}
                onChange={() => handleNotificationToggle('email')}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="email-notifications" className="ml-2 block text-sm text-gray-900">
                Email Notifications
              </label>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="slack-notifications"
                checked={notificationSettings.slack}
                onChange={() => handleNotificationToggle('slack')}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="slack-notifications" className="ml-2 block text-sm text-gray-900">
                Slack Notifications
              </label>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="webhook-notifications"
                checked={notificationSettings.webhook}
                onChange={() => handleNotificationToggle('webhook')}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="webhook-notifications" className="ml-2 block text-sm text-gray-900">
                Webhook Notifications
              </label>
            </div>
          </div>
        </div>
        
        <div className="flex items-center justify-between pt-4">
          {saveSuccess && (
            <span className="text-green-600">Configuration saved successfully!</span>
          )}
          
          <button
            type="submit"
            disabled={isSaving}
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {isSaving ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default AlertConfig;