import { useState } from 'react';
import './Sidebar.css';

const NAV_GROUPS = [
  {
    title: 'TRAFFIC AI',
    items: [
      { id: 'dashboard', label: 'Dashboard & Forecast Map', icon: '📊' },
    ],
  },
  {
    title: 'MACHINE LEARNING',
    items: [
      { id: 'models', label: 'Model Performance', icon: '🧠' },
      { id: 'training', label: 'Model Training', icon: '⚙️' },
      { id: 'evaluation', label: 'Evaluation & Metrics', icon: '📈' },
    ],
  },
  {
    title: 'DATA INTEGRATION',
    items: [
      { id: 'datasets', label: 'Datasets & Multi-Fusion', icon: '🗃️' },
    ],
  },
  {
    title: 'PROJECT INFO',
    items: [
      { id: 'about', label: 'Research & Architecture', icon: 'ℹ️' },
    ],
  },
];

export default function Sidebar({ currentPage, onNavigate, isOpen, onToggle, systemHealth, demoMode }) {
  return (
    <>
      <button className="mobile-menu-btn" onClick={onToggle}>☰</button>
      <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
        {/* Brand Header */}
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">🚦</div>
          <div className="sidebar-brand-text">
            <h2>TrafficAI</h2>
            <span>Smart City ITS Platform</span>
          </div>
        </div>

        {/* Grouped Navigation */}
        <nav className="sidebar-nav">
          {NAV_GROUPS.map((group, gIdx) => (
            <div key={gIdx} className="nav-group">
              <div className="nav-group-title">{group.title}</div>
              {group.items.map(item => (
                <button
                  key={item.id}
                  className={`sidebar-nav-item ${currentPage === item.id ? 'active' : ''}`}
                  onClick={() => onNavigate(item.id)}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span className="nav-label">{item.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        {/* Footer System Status */}
        <div className="sidebar-footer">
          <div className="sidebar-status-row">
            <span className={`status-dot ${systemHealth ? 'online' : 'offline'}`}></span>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.75rem' }}>
              {systemHealth ? 'System Operational' : 'Backend Connecting...'}
            </span>
          </div>

          <div className="sidebar-status-row demo">
            <span className="status-dot demo"></span>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a78bfa' }}>
              {demoMode ? 'Dataset Mode Active' : 'Live Data Mode'}
            </span>
          </div>

          <div className="sidebar-model-badge">
            <span>Primary Model</span>
            <strong>STGCN</strong>
          </div>
        </div>
      </aside>
      {isOpen && window.innerWidth <= 768 && (
        <div className="sidebar-overlay" onClick={onToggle}></div>
      )}
    </>
  );
}
