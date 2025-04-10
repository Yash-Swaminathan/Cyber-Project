import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, AlertTriangle, Activity, Settings } from 'lucide-react';

function Sidebar() {
  return (
    <aside className="bg-gray-800 text-gray-200 h-full w-16 md:w-64 fixed top-16 left-0 bottom-0 py-4 overflow-y-auto transition-all duration-300">
      <nav className="flex flex-col space-y-2 px-2">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `flex items-center p-3 rounded-lg transition-colors duration-200 ${
              isActive ? 'bg-blue-700 text-white' : 'hover:bg-gray-700'
            }`
          }
        >
          <Home size={20} />
          <span className="ml-3 hidden md:inline-block">Dashboard</span>
        </NavLink>

        <NavLink
          to="/alerts"
          className={({ isActive }) =>
            `flex items-center p-3 rounded-lg transition-colors duration-200 ${
              isActive ? 'bg-blue-700 text-white' : 'hover:bg-gray-700'
            }`
          }
        >
          <AlertTriangle size={20} />
          <span className="ml-3 hidden md:inline-block">Alerts</span>
        </NavLink>

        <NavLink
          to="/network"
          className={({ isActive }) =>
            `flex items-center p-3 rounded-lg transition-colors duration-200 ${
              isActive ? 'bg-blue-700 text-white' : 'hover:bg-gray-700'
            }`
          }
        >
          <Activity size={20} />
          <span className="ml-3 hidden md:inline-block">Network</span>
        </NavLink>

        <NavLink
          to="/config"
          className={({ isActive }) =>
            `flex items-center p-3 rounded-lg transition-colors duration-200 ${
              isActive ? 'bg-blue-700 text-white' : 'hover:bg-gray-700'
            }`
          }
        >
          <Settings size={20} />
          <span className="ml-3 hidden md:inline-block">Config</span>
        </NavLink>
      </nav>
    </aside>
  );
}

export default Sidebar;
