import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import { AppContext } from '../../context/AppContext';

function Header() {
  const { systemHealth } = useContext(AppContext);
  
  return (
    <header className="bg-gray-800 text-white p-4 flex justify-between items-center">
      <div className="flex items-center">
        <h1 className="text-xl font-bold">Network Anomaly Detection</h1>
      </div>
      
      <nav className="hidden md:block">
        <ul className="flex space-x-6">
          <li><Link to="/" className="hover:text-blue-300">Dashboard</Link></li>
          <li><Link to="/alerts" className="hover:text-blue-300">Alerts</Link></li>
          <li><Link to="/network" className="hover:text-blue-300">Network</Link></li>
          <li><Link to="/config" className="hover:text-blue-300">Configuration</Link></li>
        </ul>
      </nav>
      
      <div className="flex items-center space-x-3">
        <div className="flex items-center">
          <span className="mr-2">Status:</span>
          <span className={`h-3 w-3 rounded-full ${systemHealth ? 'bg-green-500' : 'bg-red-500'}`}></span>
        </div>
      </div>
    </header>
  );
}

export default Header;