import { AnimatePresence, motion } from 'motion/react';
import {
  LayoutDashboard,
  Brain,
  Cpu,
  BarChart2,
  Database,
  ChevronLeft,
  ChevronRight,
  Activity,
  Wifi,
  WifiOff,
} from 'lucide-react';
import './Sidebar.css';

const NAV_GROUPS = [
  {
    title: 'TRAFFIC AI',
    items: [
      { id: 'dashboard', label: 'Dashboard & Forecast Map', Icon: LayoutDashboard },
    ],
  },
  {
    title: 'MACHINE LEARNING',
    items: [
      { id: 'models',     label: 'Model Performance',     Icon: Brain     },
      { id: 'training',   label: 'Model Training',        Icon: Cpu       },
      { id: 'evaluation', label: 'Evaluation & Metrics',  Icon: BarChart2 },
    ],
  },
  {
    title: 'DATA INTEGRATION',
    items: [
      { id: 'datasets', label: 'Datasets & Multi-Fusion', Icon: Database },
    ],
  },
];

export default function Sidebar({ currentPage, onNavigate, isOpen, onToggle, systemHealth, demoMode }) {
  return (
    <>
      {/* Mobile hamburger */}
      <button className="mobile-menu-btn" onClick={onToggle} aria-label="Toggle menu">
        {isOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
      </button>

      {/* Sidebar panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.aside
            className="sidebar"
            initial={{ x: -264 }}
            animate={{ x: 0 }}
            exit={{ x: -264 }}
            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
          >
            {/* Brand header */}
            <div className="sidebar-brand">
              <div className="sidebar-brand-icon">
                <Activity size={22} strokeWidth={2.5} />
              </div>
              <div className="sidebar-brand-text">
                <h2>TrafficAI</h2>
                <span>Smart City ITS Platform</span>
              </div>
            </div>

            {/* Navigation */}
            <nav className="sidebar-nav">
              {NAV_GROUPS.map((group, gIdx) => (
                <div key={gIdx} className="nav-group">
                  <div className="nav-group-title">{group.title}</div>
                  {group.items.map(({ id, label, Icon }) => {
                    const isActive = currentPage === id;
                    return (
                      <motion.button
                        key={id}
                        className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                        onClick={() => onNavigate(id)}
                        whileHover={{ x: 2 }}
                        whileTap={{ scale: 0.98 }}
                        transition={{ duration: 0.15 }}
                      >
                        {/* Active indicator bar */}
                        {isActive && (
                          <motion.span
                            className="nav-active-bar"
                            layoutId="nav-active-bar"
                            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
                          />
                        )}
                        <span className="nav-icon">
                          <Icon size={16} strokeWidth={isActive ? 2.5 : 2} />
                        </span>
                        <span className="nav-label">{label}</span>
                      </motion.button>
                    );
                  })}
                </div>
              ))}
            </nav>

            {/* Footer */}
            <div className="sidebar-footer">
              <div className="sidebar-status-row">
                {systemHealth ? (
                  <Wifi size={13} className="status-icon online" />
                ) : (
                  <WifiOff size={13} className="status-icon offline" />
                )}
                <span className={`status-text ${systemHealth ? 'online' : 'offline'}`}>
                  {systemHealth ? 'System Operational' : 'Backend Connecting…'}
                </span>
              </div>

              <div className="sidebar-status-row">
                <span className="status-dot demo" />
                <span className="status-text demo">
                  {demoMode ? 'Dataset Mode Active' : 'Live Data Mode'}
                </span>
              </div>

              <div className="sidebar-model-badge">
                <span className="model-badge-label">Primary Model</span>
                <strong className="model-badge-value">STGCN</strong>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Mobile backdrop */}
      <AnimatePresence>
        {isOpen && typeof window !== 'undefined' && window.innerWidth <= 768 && (
          <motion.div
            className="sidebar-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onToggle}
          />
        )}
      </AnimatePresence>
    </>
  );
}
