import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
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
        <BrowserRouter>
          <div className="flex flex-col min-h-screen">
            <Header />
            <div className="flex flex-1 pt-16">
              <Sidebar />
              <main className="flex-1 p-4 ml-16 md:ml-64">
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
        </BrowserRouter>
      </APIProvider>
    </AppProvider>
  );
}

export default App;
