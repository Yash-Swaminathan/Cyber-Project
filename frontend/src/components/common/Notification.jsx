import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';

function Notification({ message, type = 'info', duration = 5000, onClose }) {
  const [isVisible, setIsVisible] = useState(true);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onClose, 300); // Allow animation to complete
    }, duration);
    
    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const bgColor = {
    info: 'bg-blue-500',
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
  }[type] || 'bg-blue-500';
  
  return (
    <div className={`fixed top-4 right-4 z-50 max-w-md transform transition-all duration-300 ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}`}>
      <div className={`${bgColor} text-white px-4 py-3 rounded shadow-lg flex items-center justify-between`}>
        <div className="mr-4">
          {message}
        </div>
        <button onClick={() => setIsVisible(false)} className="text-white hover:text-gray-200">
          <X size={18} />
        </button>
      </div>
    </div>
  );
}

export default Notification;