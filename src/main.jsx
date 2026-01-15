// src/main.jsx
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import { DatasetProvider } from './context/DatasetContext';

// 1. Import the necessary AG Grid modules
import { ModuleRegistry } from 'ag-grid-community';
import { AllEnterpriseModule } from 'ag-grid-enterprise';

// 2. Register the modules correctly
ModuleRegistry.registerModules([AllEnterpriseModule]);


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <DatasetProvider>
      <App />
    </DatasetProvider>
  </StrictMode>,
);