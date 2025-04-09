import { useState, useEffect, useRef } from 'react';

/**
 * A custom hook that polls an API endpoint at a specified interval
 * 
 * @param {Function} fetchFunction - The function to call for fetching data
 * @param {number} interval - The polling interval in milliseconds
 * @param {boolean} startImmediately - Whether to start polling immediately
 * @param {Array} dependencies - Dependencies that will trigger a refetch when changed
 * @returns {Object} - The data, loading state, error, and control functions
 */
function usePolling(fetchFunction, interval = 5000, startImmediately = true, dependencies = []) {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(startImmediately);
  
  const intervalRef = useRef(null);
  const abortControllerRef = useRef(null);
  
  const fetchData = async () => {
    // Cancel any in-flight requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create a new abort controller for this request
    abortControllerRef.current = new AbortController();
    
    try {
      setIsLoading(true);
      const result = await fetchFunction(abortControllerRef.current.signal);
      setData(result);
      setError(null);
    } catch (err) {
      // Only set error if not aborted
      if (err.name !== 'AbortError') {
        setError(err.message || 'An error occurred');
        console.error('Polling error:', err);
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  const startPolling = () => {
    setIsPolling(true);
  };
  
  const stopPolling = () => {
    setIsPolling(false);
  };
  
  const refetch = () => {
    fetchData();
  };
  
  useEffect(() => {
    if (isPolling) {
      // Fetch immediately
      fetchData();
      
      // Set up interval
      intervalRef.current = setInterval(fetchData, interval);
      
      // Clean up on unmount or when dependencies change
      return () => {
        clearInterval(intervalRef.current);
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
      };
    } else {
      // Clean up interval if polling is stopped
      clearInterval(intervalRef.current);
    }
  }, [isPolling, interval, ...dependencies]);
  
  return {
    data,
    isLoading,
    error,
    startPolling,
    stopPolling,
    refetch
  };
}

export default usePolling;