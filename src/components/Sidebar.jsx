// src/components/Sidebar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import { LuLayoutDashboard, LuUpload, LuTable, LuTrendingUp } from 'react-icons/lu';
import { FaFileCsv } from 'react-icons/fa';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <FaFileCsv size={25} className="logo-icon" />
        <h2>GraphIQ</h2>
      </div>
      <nav className="sidebar-nav">
        <NavLink to="/dashboard"><LuLayoutDashboard /> Dashboard</NavLink>
        <NavLink to="/upload"><LuUpload /> Upload</NavLink>
        <NavLink to="/data-table"><LuTable /> Data Table</NavLink>
      </nav>
      <div className="sidebar-pro-tip">
        <span role="img" aria-label="light-bulb">💡</span>
        <div>
          <p><strong>Pro Tip</strong></p>
          <p>Upload CSV files to get started</p>
        </div>
      </div>
      <div className="sidebar-footer">
        {/* Placeholder for Quick Actions or Profile */}
      </div>
    </div>
  );
};

export default Sidebar;