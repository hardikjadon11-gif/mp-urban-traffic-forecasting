import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import About from './pages/About';
import api from './services/api';
import './App.css';

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
      case 'dashboard': return <Dashboard demoMode={demoMode} />;
      case 'datasets': return <Datasets demoMode={demoMode} />;
      case 'models': return <Models demoMode={demoMode} />;
      case 'training': return <Training demoMode={demoMode} />;
      case 'evaluation': return <Evaluation demoMode={demoMode} />;
      case 'about': return <About />;
      default: return <Dashboard demoMode={demoMode} />;
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
      <main className="app-content" style={{ marginLeft: sidebarOpen ? 'var(--sidebar-width)' : '0' }}>
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
