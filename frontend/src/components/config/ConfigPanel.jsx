import React, { useContext, useEffect, useState } from 'react';
import { AppContext } from '../../context/AppContext';
import AlertConfig from './AlertConfig';
import TestAlert from './TestAlert';

function ConfigPanel() {
  const { getConfig, updateConfig } = useContext(AppContext);
  const [config, setConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const loadConfig = async () => {
      try {
        setIsLoading(true);
        const data = await getConfig();
        setConfig(data);
        setError(null);
      } catch (err) {
        setError('Failed to load configuration');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadConfig();
  }, [getConfig]);
  
  const handleSaveConfig = async (newConfig) => {
    try {
      setIsLoading(true);
      await updateConfig(newConfig);
      setConfig(newConfig);
      return true;
    } catch (err) {
      setError('Failed to update configuration');
      console.error(err);
      return false;
    } finally {
      setIsLoading(false);
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
  
  return (
    <div className="p-4 space-y-6">
      <h1 className="text-2xl font-bold">System Configuration</h1>
      
      {config && (
        <>
          <AlertConfig 
            config={config.alerts} 
            onSave={(alertConfig) => {
              const newConfig = {...config, alerts: alertConfig};
              return handleSaveConfig(newConfig);
            }} 
          />
          
          <TestAlert />
        </>
      )}
    </div>
  );
}

export default ConfigPanel;