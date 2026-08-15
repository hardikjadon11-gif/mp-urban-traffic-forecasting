import React from 'react';

export default function About() {
  const researchGaps = [
    {
      num: '01',
      title: 'Geographic Bias',
      desc: 'Existing benchmarks (METR-LA, PeMS-BAY) focus almost exclusively on structured US highways. They are rarely evaluated on mixed, irregular traffic networks in developing regions like Indian cities.'
    },
    {
      num: '02',
      title: 'Limited Data Fusion',
      desc: 'Many existing traffic models use single-source sensor feeds. They do not combine loop-detector speed data, zonal travel times, and spatial graphs into a unified spatio-temporal dataset.'
    },
    {
      num: '03',
      title: 'Deployment Challenge',
      desc: 'Heavy GNN and Transformer models achieve high benchmark accuracy but suffer high computational latency, hindering real-time urban ITS deployment. Lightweight-first benchmarking is required.'
    }
  ];

  const architectureFlow = [
    { stage: 'DATA SOURCES', sub: 'METR-LA • PeMS-BAY • Uber Movement • India MP Network' },
    { stage: 'DATA FUSION', sub: 'Temporal & Spatial Alignment • Normalization • Weighted Fusion' },
    { stage: 'PREPROCESSING', sub: 'Missing Value Imputation • Resampling • Outlier Cleaning' },
    { stage: 'FEATURE ENGINEERING', sub: 'Lag Steps (t-1..12) • Temporal Cyclical Features • Spatial Graph' },
    { stage: 'MODEL LAYER', sub: 'XGBoost (Baseline) | LSTM (Secondary) | STGCN (Primary Model)' },
    { stage: 'FORECAST HORIZONS', sub: '30-Minute & 60-Minute Predictive Traffic Speeds' },
    { stage: 'VISUALIZATION & ANALYTICS', sub: 'Interactive Map Dashboard • Speed Trend Line Chart' },
    { stage: 'EVALUATION & METRICS', sub: 'MAE • RMSE • MAPE Evaluation Pipeline' },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Research Context & System Architecture</h1>
        <p>Smart Traffic Congestion Forecasting System for Urban Mobility Management</p>
      </div>

      {/* Core Objective Summary */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)', background: 'linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(26,31,53,0.9) 100%)' }}>
        <h2 style={{ fontSize: '1.25rem', marginBottom: 12, color: 'var(--accent-primary)' }}>Core Project Objective</h2>
        <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '0.95rem' }}>
          Shift traffic management from <strong style={{ color: 'var(--moderate)' }}>reactive traffic response</strong> (responding after congestion forms) to <strong style={{ color: 'var(--free-flow)' }}>proactive, forecast-driven management</strong> by predicting short-term traffic congestion for individual road segments using multi-source data fusion and Spatio-Temporal Graph Convolutional Networks (STGCN).
        </p>
      </div>

      {/* Section 13: Research Gap ("Why This Project?") */}
      <div style={{ marginBottom: 'var(--space-xl)' }}>
        <h2 style={{ fontSize: '1.25rem', marginBottom: 6, color: 'var(--accent-primary)' }}>Why This Project? (Research Gap)</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: 16 }}>
          Identified gaps in current Intelligent Transportation System (ITS) literature
        </p>

        <div className="grid-3" style={{ gap: 'var(--space-md)' }}>
          {researchGaps.map(gap => (
            <div key={gap.num} className="card">
              <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-primary)', opacity: 0.6, marginBottom: 4 }}>
                {gap.num}
              </div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>
                {gap.title}
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.83rem', lineHeight: 1.6 }}>
                {gap.desc}
              </p>
            </div>
          ))}
        </div>

        {/* Our Approach Card */}
        <div className="card" style={{ marginTop: 'var(--space-md)', background: 'var(--accent-primary-dim)', borderColor: 'rgba(0, 212, 255, 0.3)' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-primary)', marginBottom: 6 }}>
            OUR APPROACH
          </h3>
          <p style={{ color: 'var(--text-primary)', fontSize: '0.88rem', lineHeight: 1.6 }}>
            Combine multi-source traffic data fusion (loop detectors + travel times) with lightweight-first model benchmarking (XGBoost baseline → LSTM RNN → STGCN primary model) evaluated across both standard highways and Indian city road corridors (Bhopal, Indore, Ujjain).
          </p>
        </div>
      </div>

      {/* Section 18: System Architecture Page */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div>
            <h2 style={{ fontSize: '1.25rem', color: 'var(--accent-primary)', margin: 0 }}>System Architecture</h2>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              End-to-end data processing, neural modeling, and visualization workflow
            </div>
          </div>
          <span className="badge badge-primary">8 Layers</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, margin: '16px 0' }}>
          {architectureFlow.map((step, idx) => (
            <div key={idx} style={{ textAlign: 'center', width: '100%', maxWidth: 580 }}>
              <div style={{
                background: idx === 4 ? 'var(--accent-primary-dim)' : 'var(--bg-secondary)',
                border: idx === 4 ? '1px solid var(--accent-primary)' : '1px solid var(--border-primary)',
                padding: '14px 20px',
                borderRadius: 'var(--radius-md)',
              }}>
                <div style={{ fontWeight: 800, color: idx === 4 ? 'var(--accent-primary)' : 'var(--text-primary)', fontSize: '0.88rem', letterSpacing: '0.05em' }}>
                  {step.stage}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
                  {step.sub}
                </div>
              </div>
              {idx < architectureFlow.length - 1 && (
                <div style={{ color: 'var(--accent-primary)', fontSize: '1rem', padding: '2px 0' }}>↓</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Tech Stack Summary */}
      <div className="grid-2">
        <div className="card">
          <h3 style={{ fontSize: '1rem', marginBottom: 12, color: 'var(--accent-primary)' }}>Backend Frameworks</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {['Python 3.10+ & FastAPI', 'PyTorch (STGCN & LSTM)', 'XGBoost Machine Learning', 'NumPy, Pandas, SciPy', 'scikit-learn Metrics', 'SQLite & SQLAlchemy ORM'].map(item => (
              <li key={item} style={{ padding: '4px 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>▸ {item}</li>
            ))}
          </ul>
        </div>
        <div className="card">
          <h3 style={{ fontSize: '1rem', marginBottom: 12, color: 'var(--accent-primary)' }}>Frontend Stack</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {['React 18 & Vite 6', 'Leaflet Interactive Maps', 'Vanilla CSS Design System', 'Inter (Google Typography)', 'Responsive Glassmorphism UI'].map(item => (
              <li key={item} style={{ padding: '4px 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>▸ {item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
