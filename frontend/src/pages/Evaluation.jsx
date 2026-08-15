import { useState, useEffect } from 'react';
import api from '../services/api';

export default function Evaluation({ demoMode }) {
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getModelPerformance()
      .then(data => { setPerformance(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div><div className="page-header"><h1>Evaluation & Validation Strategy</h1></div><div className="loading-spinner"></div></div>;

  const models = performance?.models || [];
  const metrics = [
    {
      key: 'mae',
      name: 'MAE',
      full: 'Mean Absolute Error',
      desc: 'Measures average magnitude of prediction errors. Lower is better. Computed as the mean of |actual - predicted| across test samples.',
      formula: 'MAE = (1/n) Σ |yᵢ - ŷᵢ|'
    },
    {
      key: 'rmse',
      name: 'RMSE',
      full: 'Root Mean Squared Error',
      desc: 'Penalizes larger prediction errors more heavily than MAE. Square root of the mean squared difference. Sensitive to extreme congestion outliers.',
      formula: 'RMSE = √((1/n) Σ (yᵢ - ŷᵢ)²)'
    },
    {
      key: 'mape',
      name: 'MAPE',
      full: 'Mean Absolute Percentage Error',
      desc: 'Expresses error as a percentage of actual traffic speed. Useful for comparing across different speed scales (highway vs urban road).',
      formula: 'MAPE = (100/n) Σ |(yᵢ - ŷᵢ) / yᵢ|'
    },
  ];

  const validationSteps = [
    { step: '1', title: 'Chronological Train/Test Split (70/15/15)', desc: 'Data is strictly split chronologically (70% train / 15% validation / 15% test). No random shuffling is performed, preserving temporal sequence order and preventing data leakage.' },
    { step: '2', title: 'Naive Last-Known-Value Baseline', desc: 'A naive baseline model (predicting the last observed speed t for time t+k) is evaluated as a benchmark. ML models must demonstrate clear performance improvements over naive persistence.' },
    { step: '3', title: 'Identical Test Window Evaluation', desc: 'XGBoost (Baseline), LSTM (Secondary), and STGCN (Primary) are evaluated on the exact same test sequence window for fair, standardized comparison.' },
    { step: '4', title: 'Non-Leaky Data Normalization', desc: 'Normalization parameters (MinMax scaler) are fitted strictly on the training set. Validation and test sets are transformed using training statistics only.' },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Evaluation Methodology & Validation Strategy</h1>
        <p>Evaluation metrics, error formulas, and rigorous non-leaky validation procedures</p>
      </div>

      {demoMode && (
        <div className="demo-banner">
          <strong>⚡ DEMO MODE</strong>
          <span>Train models via the Model Training page to compute live MAE, RMSE, and MAPE scores.</span>
        </div>
      )}

      {/* Metric Cards (Section 14) */}
      <h2 style={{ fontSize: '1.2rem', marginBottom: 16, color: 'var(--accent-primary)' }}>Performance Metrics</h2>
      <div className="grid-3" style={{ marginBottom: 'var(--space-xl)' }}>
        {metrics.map(m => (
          <div key={m.key} className="card">
            <div className="card-header">
              <span className="card-title">{m.name}</span>
            </div>
            <div style={{ color: 'var(--accent-primary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>{m.full}</div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.83rem', lineHeight: 1.6, marginBottom: 12 }}>{m.desc}</p>
            <div style={{ background: 'var(--bg-secondary)', padding: '10px 14px', borderRadius: 'var(--radius-md)', fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {m.formula}
            </div>

            {/* Model Values for this metric */}
            <div style={{ marginTop: 16 }}>
              {models.map(model => (
                <div key={model.model_name} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border-primary)' }}>
                  <span style={{ fontSize: '0.8rem', color: model.is_primary ? 'var(--accent-primary)' : 'var(--text-secondary)', fontWeight: model.is_primary ? 700 : 500 }}>
                    {model.model_name.toUpperCase()} {model.is_primary ? '(Primary)' : ''}
                  </span>
                  <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>
                    {model[m.key] != null ? (m.key === 'mape' ? `${model[m.key].toFixed(2)}%` : model[m.key].toFixed(4)) : <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>Awaiting eval</span>}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Validation Strategy Section (Section 14) */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div>
            <h2 style={{ fontSize: '1.2rem', color: 'var(--accent-primary)', margin: 0 }}>Validation Strategy</h2>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              4-step chronological validation methodology for time-series forecasting
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 12 }}>
          {validationSteps.map(item => (
            <div key={item.step} className="pipeline-step" style={{ background: 'var(--bg-secondary)', padding: '14px 18px', borderRadius: 'var(--radius-md)' }}>
              <span className="pipeline-step-number">{item.step}</span>
              <div>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.9rem' }}>{item.title}</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', lineHeight: 1.5, marginTop: 4 }}>{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chronological Data Split Visualization */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Chronological Dataset Split Visualization</span>
        </div>
        <div style={{ display: 'flex', height: 42, borderRadius: 'var(--radius-md)', overflow: 'hidden', marginBottom: 12 }}>
          <div style={{ flex: 70, background: 'rgba(0, 212, 255, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
            Training Split (70%)
          </div>
          <div style={{ flex: 15, background: 'rgba(167, 139, 250, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700, color: '#a78bfa' }}>
            Validation (15%)
          </div>
          <div style={{ flex: 15, background: 'rgba(245, 158, 11, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700, color: 'var(--moderate)' }}>
            Test Split (15%)
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>← Earlier Timestamps (Historical Data)</span>
          <span>Future Timestamps (Test Evaluation) →</span>
        </div>
      </div>
    </div>
  );
}
