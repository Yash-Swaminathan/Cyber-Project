import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, AlertTriangle, Activity, Settings } from 'lucide-react';

function Sidebar() {
  return (
    <div className="bg-gray-900 text-white h-full w-16 md:w-64 fixed left-0 top-16 bottom-0 py-4">
      <div className="flex flex-col h-full">
        <nav className="flex-1">
          <ul className="space-y-2 px-2">
            <li>
              <NavLink 
                to="/" 
                className={({ isActive }) => 
                  `flex items-center p-2 rounded-lg ${isActive ? 'bg-blue-700' : 'hover:bg-gray-700'}`
                }
                end
              >
                <Home size={20} />
                <span className="ml-3 hidden md:block">Dashboard</span>
              </NavLink>
            </li>
            <li>
              <NavLink 
                to="/alerts" 
                className={({ isActive }) => 
                  `flex items-center p-2 rounded-lg ${isActive ? 'bg-blue-700' : 'hover:bg-gray-700'}`
                }
              >
                <AlertTriangle size={20} />
                <span className="ml-3 hidden md:block">Alerts</span>
              </NavLink>
            </li>
            <li>
              <NavLink 
                to="/network" 
                className={({ isActive }) => 
                  `flex items-center p-2 rounded-lg ${isActive ? 'bg-blue-700' : 'hover:bg-gray-700'}`
                }
              >
                <Activity size={20} />
                <span className="ml-3 hidden md:block">Network</span>
              </NavLink>
            </li>
            <li>
              <NavLink 
                to="/config" 
                className={({ isActive }) => 
                  `flex items-center p-2 rounded-lg ${isActive ? 'bg-blue-700' : 'hover:bg-gray-700'}`
                }
              >
                <Settings size={20} />
                <span className="ml-3 hidden md:block">Configuration</span>
              </NavLink>
            </li>
          </ul>
        </nav>
      </div>
    </div>
  );
}

export default Sidebar;