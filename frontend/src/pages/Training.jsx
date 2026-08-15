import { useState, useEffect } from 'react';
import api from '../services/api';

export default function Training({ demoMode }) {
  const [config, setConfig] = useState({
    model_name: 'stgcn',
    dataset: 'metr-la',
    forecast_horizon: 30,
    epochs: 30,
    batch_size: 32,
    learning_rate: 0.001,
    n_sensors: 20,
  });
  const [trainingStatus, setTrainingStatus] = useState({});
  const [isTraining, setIsTraining] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    const fetchStatus = () => {
      api.getTrainingStatus()
        .then(data => {
          setTrainingStatus(data.training_status || {});
          const anyRunning = Object.values(data.training_status || {}).some(s => s.status === 'running' || s.status === 'starting');
          setIsTraining(anyRunning);
        })
        .catch(() => {});
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const startTraining = async () => {
    try {
      setMessage(null);
      const result = await api.startTraining(config);
      setMessage({ type: 'success', text: result.message });
      setIsTraining(true);
    } catch (e) {
      setMessage({ type: 'error', text: e.message });
    }
  };

  const modelStatus = trainingStatus[config.model_name] || {};

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Model Training</h1>
        <p>Configure and train XGBoost, LSTM, or STGCN models</p>
      </div>

      {demoMode && (
        <div className="demo-banner">
          <strong>⚡ DEMO MODE</strong>
          <span>Training uses demo data. For real results, provide actual METR-LA / PeMS-BAY datasets.</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
        {/* Configuration */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Training Configuration</span>
          </div>

          <div className="form-group">
            <label className="form-label">Model</label>
            <select className="form-select" value={config.model_name} onChange={e => setConfig({...config, model_name: e.target.value})}>
              <option value="xgboost">XGBoost — Classical ML Baseline</option>
              <option value="lstm">LSTM — Deep Learning Baseline</option>
              <option value="stgcn">STGCN — Primary Model (Graph + Temporal)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Dataset</label>
            <select className="form-select" value={config.dataset} onChange={e => setConfig({...config, dataset: e.target.value})}>
              <option value="metr-la">METR-LA (207 sensors, LA County)</option>
              <option value="pems-bay">PeMS-BAY (325 sensors, Bay Area)</option>
            </select>
          </div>

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label">Forecast Horizon</label>
              <select className="form-select" value={config.forecast_horizon} onChange={e => setConfig({...config, forecast_horizon: parseInt(e.target.value)})}>
                <option value="30">30 minutes</option>
                <option value="60">60 minutes</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Sensors (subset)</label>
              <input type="number" className="form-input" value={config.n_sensors} min={5} max={207}
                onChange={e => setConfig({...config, n_sensors: parseInt(e.target.value) || 20})} />
            </div>
          </div>

          {config.model_name !== 'xgboost' && (
            <div className="grid-3" style={{ gap: 8 }}>
              <div className="form-group">
                <label className="form-label">Epochs</label>
                <input type="number" className="form-input" value={config.epochs} min={1} max={500}
                  onChange={e => setConfig({...config, epochs: parseInt(e.target.value) || 50})} />
              </div>
              <div className="form-group">
                <label className="form-label">Batch Size</label>
                <input type="number" className="form-input" value={config.batch_size} min={4} max={256}
                  onChange={e => setConfig({...config, batch_size: parseInt(e.target.value) || 32})} />
              </div>
              <div className="form-group">
                <label className="form-label">Learning Rate</label>
                <input type="number" className="form-input" value={config.learning_rate} step={0.0001}
                  onChange={e => setConfig({...config, learning_rate: parseFloat(e.target.value) || 0.001})} />
              </div>
            </div>
          )}

          <button className="btn btn-primary btn-lg" onClick={startTraining}
            disabled={modelStatus.status === 'running' || modelStatus.status === 'starting'}
            style={{ width: '100%', marginTop: 8, justifyContent: 'center' }}>
            {modelStatus.status === 'running' ? '⏳ Training in Progress...' : '🚀 Start Training'}
          </button>

          {message && (
            <div style={{ marginTop: 12, padding: '10px 14px', borderRadius: 'var(--radius-md)',
              background: message.type === 'error' ? 'var(--congested-dim)' : 'var(--free-flow-dim)',
              color: message.type === 'error' ? 'var(--congested)' : 'var(--free-flow)',
              fontSize: '0.85rem' }}>
              {message.text}
            </div>
          )}
        </div>

        {/* Status Panel */}
        <div>
          {/* Current training status */}
          <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
            <div className="card-header">
              <span className="card-title">Training Status</span>
              <span className={`badge ${modelStatus.status === 'completed' ? 'badge-success' : modelStatus.status === 'running' ? 'badge-warning' : modelStatus.status === 'failed' ? 'badge-danger' : 'badge-demo'}`}>
                {modelStatus.status || 'Not started'}
              </span>
            </div>

            {(modelStatus.status === 'running' || modelStatus.status === 'starting') && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Progress</span>
                  <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>{modelStatus.progress || 0}%</span>
                </div>
                <div className="progress-bar" style={{ height: 8 }}>
                  <div className="progress-bar-fill" style={{ width: `${modelStatus.progress || 0}%` }}></div>
                </div>
              </div>
            )}

            {modelStatus.started_at && (
              <div className="detail-item" style={{ marginBottom: 8 }}>
                <span className="detail-label">Started</span>
                <span style={{ fontSize: '0.85rem' }}>{modelStatus.started_at}</span>
              </div>
            )}
            {modelStatus.completed_at && (
              <div className="detail-item" style={{ marginBottom: 8 }}>
                <span className="detail-label">Completed</span>
                <span style={{ fontSize: '0.85rem' }}>{modelStatus.completed_at}</span>
              </div>
            )}
            {modelStatus.error && (
              <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-md)', background: 'var(--congested-dim)', color: 'var(--congested)', fontSize: '0.8rem', marginTop: 8 }}>
                Error: {modelStatus.error}
              </div>
            )}
          </div>

          {/* Results */}
          {modelStatus.result?.metrics && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">Evaluation Results</span>
              </div>
              <div className="grid-3" style={{ gap: 8 }}>
                <div className="metric-card card" style={{ padding: 16 }}>
                  <div className="metric-value">{modelStatus.result.metrics.mae?.toFixed(4)}</div>
                  <div className="metric-label">MAE</div>
                </div>
                <div className="metric-card card" style={{ padding: 16 }}>
                  <div className="metric-value">{modelStatus.result.metrics.rmse?.toFixed(4)}</div>
                  <div className="metric-label">RMSE</div>
                </div>
                <div className="metric-card card" style={{ padding: 16 }}>
                  <div className="metric-value">{modelStatus.result.metrics.mape?.toFixed(2)}%</div>
                  <div className="metric-label">MAPE</div>
                </div>
              </div>
              <div style={{ marginTop: 12, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Training time: {modelStatus.result.training_time_seconds}s
                {modelStatus.result.epochs && ` • ${modelStatus.result.epochs} epochs`}
              </div>
            </div>
          )}

          {/* All Models Status */}
          <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
            <div className="card-header">
              <span className="card-title">All Models</span>
            </div>
            {['xgboost', 'lstm', 'stgcn'].map(name => {
              const s = trainingStatus[name] || {};
              return (
                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--border-primary)' }}>
                  <span style={{ fontWeight: 600, color: name === 'stgcn' ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
                    {name.toUpperCase()}
                    {name === 'stgcn' && <span className="badge badge-primary" style={{ marginLeft: 6 }}>Primary</span>}
                  </span>
                  <span className={`badge ${s.status === 'completed' ? 'badge-success' : s.status === 'running' ? 'badge-warning' : 'badge-demo'}`} style={{ fontSize: '0.7rem' }}>
                    {s.status || 'Not trained'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
