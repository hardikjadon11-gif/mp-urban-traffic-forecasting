import { useState, useEffect } from 'react';
import api from '../services/api';

export default function Models({ demoMode }) {
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getModelPerformance()
      .then(data => { setPerformance(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div><div className="page-header"><h1>Model Performance & Architecture</h1></div><div className="loading-spinner"></div></div>;

  const models = performance?.models || [];
  const modelRoles = {
    xgboost: { type: 'Classical ML', role: 'Baseline', badge: 'Baseline' },
    lstm: { type: 'Deep Learning', role: 'Secondary Model', badge: 'Secondary' },
    stgcn: { type: 'Spatio-Temporal Graph Neural Network', role: 'Primary Model', badge: 'PRIMARY MODEL' },
  };

  const descriptions = {
    xgboost: {
      full: 'XGBoost (Extreme Gradient Boosting)',
      desc: 'Classical machine learning baseline. Uses engineered tabular features (historical lags t-1..t-12, time-of-day, day-of-week, rolling mean/std, spatial neighbor speeds) to predict short-term traffic speeds.',
      strengths: ['Fast training & low resource usage', 'High interpretability via feature importance', 'Strong baseline for tabular feature representations']
    },
    lstm: {
      full: 'Long Short-Term Memory Network',
      desc: 'Secondary deep learning temporal baseline. Recurrent neural network architecture equipped with input, forget, and output gates to model non-linear sequential dependencies across time series.',
      strengths: ['Captures long-range temporal dependencies', 'Learns daily & weekly recurring traffic cycles', 'Handles continuous non-linear time series']
    },
    stgcn: {
      full: 'Spatio-Temporal Graph Convolutional Network (STGCN)',
      desc: 'Primary forecasting model. Integrates spatial graph convolutions (Chebyshev polynomial spectral convolution over road network graph) with 1D temporal gated convolutions (GLU) in stacked ST-Conv blocks.',
      strengths: ['Simultaneous Spatio-Temporal modeling', 'Explicit road network graph topology', 'Chebyshev spectral graph convolutions', 'Fast convergence with GLU gating & LayerNorm']
    },
  };

  const maxMAE = Math.max(...models.map(m => m.mae || 0), 1);
  const maxRMSE = Math.max(...models.map(m => m.rmse || 0), 1);

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Model Performance & Comparison</h1>
        <p>Comparative evaluation of XGBoost (Baseline), LSTM (Secondary), and STGCN (Primary Model)</p>
      </div>

      {demoMode && (
        <div className="demo-banner">
          <strong>⚡ DEMO MODE</strong>
          <span>Train models via the Model Training page to evaluate and populate real performance metrics.</span>
        </div>
      )}

      {/* Model Performance Comparison Table (Section 10) */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)', overflowX: 'auto' }}>
        <div className="card-header">
          <div>
            <span className="card-title" style={{ fontSize: '1.2rem' }}>Model Performance Comparison</span>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Evaluated on identical test windows across all models (70% Train / 15% Val / 15% Test)
            </div>
          </div>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Category</th>
              <th>Role</th>
              <th>MAE (mph)</th>
              <th>RMSE (mph)</th>
              <th>MAPE (%)</th>
              <th>Training Time</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => {
              const meta = modelRoles[m.model_name] || {};
              return (
                <tr key={m.model_name} style={{ background: m.is_primary ? 'rgba(0, 212, 255, 0.04)' : undefined }}>
                  <td>
                    <strong style={{ color: m.is_primary ? 'var(--accent-primary)' : 'var(--text-primary)', fontSize: '1rem' }}>
                      {m.model_name.toUpperCase()}
                    </strong>
                    {m.is_primary && (
                      <span className="badge badge-primary" style={{ marginLeft: 8, fontWeight: 700 }}>
                        PRIMARY MODEL
                      </span>
                    )}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>{meta.type}</td>
                  <td>
                    <span className={`badge ${m.is_primary ? 'badge-primary' : 'badge-demo'}`}>
                      {meta.role}
                    </span>
                  </td>
                  <td style={{ fontWeight: 700, color: m.mae != null ? 'var(--accent-primary)' : 'var(--text-muted)' }}>
                    {m.mae != null ? m.mae.toFixed(4) : <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>Awaiting evaluation</span>}
                  </td>
                  <td style={{ fontWeight: 700 }}>
                    {m.rmse != null ? m.rmse.toFixed(4) : <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>Awaiting evaluation</span>}
                  </td>
                  <td style={{ fontWeight: 700 }}>
                    {m.mape != null ? `${m.mape.toFixed(2)}%` : <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>Awaiting evaluation</span>}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {m.training_time_seconds != null ? `${m.training_time_seconds}s` : '—'}
                  </td>
                  <td>
                    {m.is_trained ? (
                      <span className="badge badge-success">Trained</span>
                    ) : (
                      <span className="badge" style={{ background: 'rgba(100,116,139,0.15)', color: 'var(--text-muted)' }}>Not trained</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Visual Metric Bar Charts */}
      {models.some(m => m.mae != null) && (
        <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
          <div className="card">
            <div className="card-header">
              <span className="card-title">Mean Absolute Error (MAE)</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 16 }}>Lower is better. Measures magnitude of error in mph.</p>
            {models.map(m => (
              <div key={m.model_name} style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                  <span style={{ color: m.is_primary ? 'var(--accent-primary)' : 'var(--text-secondary)', fontWeight: 600 }}>{m.model_name.toUpperCase()}</span>
                  <span style={{ fontWeight: 700 }}>{m.mae != null ? m.mae.toFixed(4) : '—'}</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill" style={{ width: m.mae != null ? `${(m.mae / maxMAE) * 100}%` : '0%' }}></div>
                </div>
              </div>
            ))}
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Root Mean Squared Error (RMSE)</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 16 }}>Lower is better. Penalizes larger prediction outliers.</p>
            {models.map(m => (
              <div key={m.model_name} style={{ marginBottom: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                  <span style={{ color: m.is_primary ? 'var(--accent-primary)' : 'var(--text-secondary)', fontWeight: 600 }}>{m.model_name.toUpperCase()}</span>
                  <span style={{ fontWeight: 700 }}>{m.rmse != null ? m.rmse.toFixed(4) : '—'}</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill" style={{ width: m.rmse != null ? `${(m.rmse / maxRMSE) * 100}%` : '0%' }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Cards */}
      <h2 style={{ fontSize: '1.2rem', marginBottom: 16, color: 'var(--accent-primary)' }}>Detailed Model Architectures</h2>
      <div className="grid-3">
        {models.map(m => {
          const desc = descriptions[m.model_name] || {};
          const meta = modelRoles[m.model_name] || {};
          return (
            <div key={m.model_name} className="card" style={{ borderColor: m.is_primary ? 'rgba(0,212,255,0.4)' : undefined, background: m.is_primary ? 'var(--bg-card)' : undefined }}>
              <div className="card-header">
                <span className="card-title" style={{ fontSize: '1.1rem' }}>{m.model_name.toUpperCase()}</span>
                <span className={`badge ${m.is_primary ? 'badge-primary' : 'badge-demo'}`}>
                  {meta.badge}
                </span>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', marginBottom: 10, fontWeight: 600 }}>{desc.full}</div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 14, lineHeight: 1.6 }}>{desc.desc}</p>
              {desc.strengths && (
                <ul style={{ listStyle: 'none', padding: 0 }}>
                  {desc.strengths.map((s, i) => (
                    <li key={i} style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '3px 0' }}>✓ {s}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
