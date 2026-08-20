import { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import api from './services/api';
import './App.css';

const PAGE_VARIANTS = {
  initial:  { opacity: 0, y: 8 },
  animate:  { opacity: 1, y: 0, transition: { duration: 0.22, ease: [0.4, 0, 0.2, 1] } },
  exit:     { opacity: 0, y: -6, transition: { duration: 0.15, ease: [0.4, 0, 0.2, 1] } },
};

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [systemHealth, setSystemHealth] = useState(null);
  const [demoMode, setDemoMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);

  useEffect(() => {
    api.health().then(data => {
      setSystemHealth(data);
      setDemoMode(data.demo_mode);
    }).catch(() => {});

    const handleResize = () => {
      setSidebarOpen(window.innerWidth > 768);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':  return <Dashboard  demoMode={demoMode} />;
      case 'datasets':   return <Datasets   demoMode={demoMode} />;
      case 'models':     return <Models     demoMode={demoMode} />;
      case 'training':   return <Training   demoMode={demoMode} />;
      case 'evaluation': return <Evaluation demoMode={demoMode} />;
      default:           return <Dashboard  demoMode={demoMode} />;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        systemHealth={systemHealth}
        demoMode={demoMode}
      />
      <main
        className="app-content"
        style={{ marginLeft: sidebarOpen ? 'var(--sidebar-width)' : '0' }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPage}
            variants={PAGE_VARIANTS}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            {renderPage()}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
