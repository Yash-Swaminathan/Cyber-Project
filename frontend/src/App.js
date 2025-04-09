import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { APIProvider } from './context/APIContext';
import Header from './components/common/Header';
import Sidebar from './components/common/Sidebar';
import Dashboard from './components/dashboard/Dashboard';
import AlertsList from './components/alerts/AlertsList';
import AlertDetails from './components/alerts/AlertDetails';
import NetworkGraph from './components/network/NetworkGraph';
import ConfigPanel from './components/config/ConfigPanel';
import './App.css';

function App() {
  return (
    <AppProvider>
      <APIProvider>
        <Router>
          <div className="flex flex-col min-h-screen bg-gray-100">
            <Header />
            <div className="flex flex-1">
              <Sidebar />
              <main className="flex-1 p-4 ml-16 md:ml-64 mt-16">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/alerts" element={<AlertsList />} />
                  <Route path="/alerts/:id" element={<AlertDetails />} />
                  <Route path="/network" element={<NetworkGraph />} />
                  <Route path="/config" element={<ConfigPanel />} />
                </Routes>
              </main>
            </div>
          </div>
        </Router>
      </APIProvider>
    </AppProvider>
  );
}

export default App;