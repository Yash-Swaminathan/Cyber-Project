import React, { useEffect, useContext, useState } from 'react';
import { AppContext } from '../../context/AppContext';
import AlertConfig from './AlertConfig';
import TestAlert from './TestAlert';

function ConfigPanel() {
  const { config, getConfig, updateConfig, isLoading, error } = useContext(AppContext);
  const [localConfig, setLocalConfig] = useState(null);

  // Load config once on mount.
  useEffect(() => {
    async function load() {
      const cfg = await getConfig();
      setLocalConfig(cfg);
    }
    load();
  }, [getConfig]);

  const handleSaveConfig = async (newConfig) => {
    try {
      const updated = await updateConfig(newConfig);
      setLocalConfig(updated);
      return true;
    } catch (err) {
      console.error(err);
      return false;
    }
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="p-4">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong className="font-bold">Error:</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      </div>
    );
  }
  
  if (!localConfig) {
    return <div className="p-4">No configuration data available.</div>;
  }

  // Provide fallback if localConfig.alerts is undefined.
  const configForAlert = localConfig.alerts || {
    thresholds: { low: 0.3, medium: 0.6, high: 0.8 },
    notifications: { email: false, slack: false, webhook: false }
  };

  return (
    <div className="p-4 space-y-6">
      <h1 className="text-2xl font-bold">System Configuration</h1>
      <AlertConfig 
        config={configForAlert} 
        onSave={(alertConfig) => {
          const newConfig = { ...localConfig, alerts: alertConfig };
          return handleSaveConfig(newConfig);
        }} 
      />
      <TestAlert />
    </div>
  );
}

export default ConfigPanel;
