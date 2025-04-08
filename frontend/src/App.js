import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Header from './components/common/Header';
import Sidebar from './components/common/Sidebar';
import Dashboard from './components/dashboard/Dashboard';
import AlertsList from './components/alerts/AlertsList';
import AlertDetails from './components/alerts/AlertDetails';
import NetworkGraph from './components/network/NetworkGraph';
import FlowTable from './components/network/FlowTable';
import ConfigPanel from './components/config/ConfigPanel';
import './App.css';

const App = () => {
  return (
    <AppProvider>
      <Router>
        <div className="app-container">
          <Header />
          <div className="main-content">
            <Sidebar />
            <div className="page-content">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/alerts" element={<AlertsList />} />
                <Route path="/alerts/:alertId" element={<AlertDetails />} />
                <Route path="/network" element={<NetworkGraph />} />
                <Route path="/flows" element={<FlowTable />} />
                <Route path="/config" element={<ConfigPanel />} />
              </Routes>
            </div>
          </div>
        </div>
      </Router>
    </AppProvider>
  );
};

export default App;