import React from 'react';

function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 h-16 flex items-center justify-center bg-gradient-to-r from-blue-600 to-indigo-600 shadow-lg z-10">
      <h1 className="text-2xl font-bold text-white">Network Anomaly Detection</h1>
    </header>
  );
}

export default Header;
