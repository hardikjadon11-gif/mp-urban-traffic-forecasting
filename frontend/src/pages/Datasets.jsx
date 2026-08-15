import { useState, useEffect } from 'react';
import api from '../services/api';

export default function Datasets({ demoMode }) {
  const [datasets, setDatasets] = useState([]);
  const [fusionStatus, setFusionStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.getDatasets(), api.getFusionStatus()])
      .then(([ds, fs]) => {
        setDatasets(ds.datasets || []);
        setFusionStatus(fs);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div><div className="page-header"><h1>Datasets & Multi-Source Data Fusion</h1></div><div className="loading-spinner"></div></div>;

  const icons = { 'metr-la': '🛣️', 'pems-bay': '🌉', 'uber': '🚗', 'india-mp': '🇮🇳' };

  // 5-Stage Data Fusion Process (Section 11)
  const fusionPipelineStages = [
    { step: 1, title: 'MULTI-SOURCE INPUTS', desc: 'METR-LA (207 sensors) + PeMS-BAY (325 sensors) + Uber Movement (50 zones) + India MP (250 roads)' },
    { step: 2, title: 'TEMPORAL ALIGNMENT', desc: 'Resample all continuous speed and travel-time streams to unified 15-minute intervals.' },
    { step: 3, title: 'SPATIAL ALIGNMENT', desc: 'Map sensor GPS coordinates and zone centroids to road network graph nodes via Haversine distance.' },
    { step: 4, title: 'NORMALIZATION', desc: 'Apply non-leaky MinMax scaling fitted strictly on training data split [0, 1].' },
    { step: 5, title: 'FEATURE CONCATENATION', desc: 'Combine temporal features (hour, day, peak), lag steps (t-1..t-12), and spatial adjacency.' },
    { step: 6, title: 'WEIGHTED COMBINATION', desc: 'Apply configurable source weights (45% METR-LA, 35% PeMS-BAY, 20% Uber).' },
    { step: 7, title: 'UNIFIED SPATIO-TEMPORAL TENSOR', desc: 'Feed 3D tensor (Batch, Nodes, Sequence) directly into STGCN primary model.' },
  ];

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Datasets & Multi-Source Data Fusion</h1>
        <p>Unified data ingestion, spatial-temporal alignment, and normalization pipeline</p>
      </div>

      {demoMode && (
        <div className="demo-banner">
          <strong>⚡ DEMO MODE</strong>
          <span>Using sample traffic datasets for METR-LA, PeMS-BAY, Uber Movement, and India MP.</span>
        </div>
      )}

      {/* Dataset Cards Grid */}
      <h2 style={{ fontSize: '1.2rem', marginBottom: 16, color: 'var(--accent-primary)' }}>Available Traffic Datasets</h2>
      <div className="grid-4" style={{ marginBottom: 'var(--space-xl)' }}>
        {datasets.map((ds, i) => (
          <div key={i} className="card">
            <div className="card-header">
              <span className="card-title" style={{ fontSize: '1rem' }}>{icons[ds.source] || '📦'} {ds.name}</span>
              {ds.is_demo && <span className="badge badge-demo">Dataset</span>}
            </div>
            {ds.description && <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: 16, lineHeight: 1.5 }}>{ds.description}</p>}
            <div className="grid-2" style={{ gap: 8 }}>
              <div className="detail-item">
                <span className="detail-label">Sensors / Nodes</span>
                <span style={{ color: 'var(--accent-primary)', fontWeight: 800, fontSize: '1.2rem' }}>{ds.n_sensors?.toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Records</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: '1.1rem' }}>{ds.n_records?.toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Frequency</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{ds.interval_minutes} min</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Missing Data</span>
                <span style={{ color: ds.missing_pct > 5 ? 'var(--moderate)' : 'var(--free-flow)', fontWeight: 600 }}>{ds.missing_pct}%</span>
              </div>
            </div>
            {ds.time_range_start && ds.time_range_start !== 'N/A' && (
              <div style={{ marginTop: 12, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Range: {ds.time_range_start?.slice(0, 10)} → {ds.time_range_end?.slice(0, 10)}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Visual Data Fusion Pipeline (Section 11) */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div>
            <span className="card-title" style={{ fontSize: '1.2rem' }}>Multi-Source Data Fusion Pipeline</span>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Conceptual 5-step integration process described in project research specification
            </div>
          </div>
          <span className="badge badge-primary">7 Stages</span>
        </div>

        {/* Source Inputs Visual Header */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center', border: '1px solid var(--border-primary)' }}>
            <div style={{ fontSize: '1.2rem' }}>🛣️</div>
            <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--accent-primary)' }}>METR-LA</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>207 Highways (5m)</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center', border: '1px solid var(--border-primary)' }}>
            <div style={{ fontSize: '1.2rem' }}>🌉</div>
            <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#a78bfa' }}>PeMS-BAY</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>325 Sensors (5m)</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center', border: '1px solid var(--border-primary)' }}>
            <div style={{ fontSize: '1.2rem' }}>🚗</div>
            <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--moderate)' }}>Uber Movement</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>50 Zones (1h)</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-md)', textAlign: 'center', border: '1px solid var(--border-primary)' }}>
            <div style={{ fontSize: '1.2rem' }}>🇮🇳</div>
            <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--free-flow)' }}>India MP Network</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>250 Roads (5m)</div>
          </div>
        </div>

        {/* Pipeline Step List */}
        {fusionPipelineStages.map((stage, i) => (
          <div key={i}>
            <div className="pipeline-step" style={{ background: 'var(--bg-secondary)', padding: '14px 18px', borderRadius: 'var(--radius-md)', borderLeft: '4px solid var(--accent-primary)' }}>
              <span className="pipeline-step-number">{stage.step}</span>
              <div>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.9rem', letterSpacing: '0.03em' }}>{stage.title}</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginTop: 2, lineHeight: 1.5 }}>{stage.description}</div>
              </div>
            </div>
            {i < fusionPipelineStages.length - 1 && <div className="pipeline-arrow" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '4px 0' }}>↓</div>}
          </div>
        ))}
      </div>

      {/* Configured Fusion Weights */}
      {fusionStatus?.configuration && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Fusion Source Weights Configuration</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: 16 }}>
            Configurable matrix weights dictating each dataset's contribution to the unified feature matrix.
          </p>
          <div className="grid-3">
            {Object.entries(fusionStatus.configuration.weights).map(([key, val]) => (
              <div key={key} className="card" style={{ textAlign: 'center', padding: '16px', background: 'var(--bg-secondary)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{key.replace('_', ' ')}</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-primary)', marginTop: 4 }}>{(val * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
