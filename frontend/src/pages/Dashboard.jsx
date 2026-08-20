import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import api from '../services/api';
import mpBoundaryData from '../data/mp_boundary.json';
import './Dashboard.css';

// Fix Leaflet default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Major Madhya Pradesh Reference Cities (Geographic Orientation)
const MP_REFERENCE_CITIES = [
  { name: 'Bhopal (Capital)', lat: 23.2599, lon: 77.4126, zoom: 12, icon: '🏛️', hasData: true },
  { name: 'Indore (Commercial Hub)', lat: 22.7196, lon: 75.8577, zoom: 12, icon: '🏙️', hasData: true },
  { name: 'Ujjain (Mahakal)', lat: 23.1765, lon: 75.7885, zoom: 13, icon: '🛕', hasData: true },
  { name: 'Gwalior (North MP)', lat: 26.2183, lon: 78.1828, zoom: 12, icon: '🏰', hasData: true },
  { name: 'Jabalpur (Mahakaushal)', lat: 23.1815, lon: 79.9864, zoom: 12, icon: '🌊', hasData: true },
  { name: 'Sagar (Central MP)', lat: 23.8388, lon: 78.7378, zoom: 12, icon: '📍', hasData: false },
  { name: 'Rewa (Vindhya)', lat: 24.5362, lon: 81.3037, zoom: 12, icon: '📍', hasData: false },
  { name: 'Satna (North-East)', lat: 24.6005, lon: 80.8322, zoom: 12, icon: '📍', hasData: false },
  { name: 'Ratlam (Malwa)', lat: 23.3315, lon: 75.0367, zoom: 12, icon: '📍', hasData: false },
  { name: 'Dewas (Malwa)', lat: 22.9676, lon: 76.0534, zoom: 12, icon: '📍', hasData: false },
];

// Predefined Regional Initial Viewports
const REGION_VIEWPORTS = {
  'india-mp': { name: 'State of Madhya Pradesh', lat: 23.50, lon: 78.50, zoom: 6.8, icon: '🇮🇳' },
  'metr-la': { name: 'All LA Region', lat: 34.05, lon: -118.30, zoom: 10, icon: '🏙️' },
  'pems-bay': { name: 'Bay Area Region', lat: 37.55, lon: -122.20, zoom: 9, icon: '🌉' },
};

// Model comparison colors
const MODEL_COLORS = {
  stgcn: { color: '#00d4ff', label: 'STGCN', dash: '' },
  lstm: { color: '#a78bfa', label: 'LSTM', dash: '6,3' },
  xgboost: { color: '#10b981', label: 'XGBoost', dash: '3,3' },
};

export default function Dashboard({ demoMode }) {
  const [datasetSource, setDatasetSource] = useState('india-mp');
  const [trafficData, setTrafficData] = useState(null);
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [horizon, setHorizon] = useState(0); // 0 = current, 30 = 30-min forecast, 60 = 60-min forecast
  const [selectedModel, setSelectedModel] = useState('stgcn');
  const [demoTimeIndex, setDemoTimeIndex] = useState(200);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeArea, setActiveArea] = useState(REGION_VIEWPORTS['india-mp']);
  const [modelPerf, setModelPerf] = useState(null);
  const [cityNotice, setCityNotice] = useState(null);
  const [showCoverageOverlay, setShowCoverageOverlay] = useState(true);

  // Feature 1: Model Comparison Toggle
  const [compareModels, setCompareModels] = useState({ stgcn: true, lstm: false, xgboost: false });
  const [multiModelData, setMultiModelData] = useState(null);

  // Feature 2: Disruption Simulator
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simDuration, setSimDuration] = useState(30);
  const simMarkersRef = useRef([]);

  // Feature 3: Feature Importance
  const [featureImportance, setFeatureImportance] = useState(null);
  const [showFeatureImportance, setShowFeatureImportance] = useState(false);

  // Feature 5: Last Updated / Live Feel
  const [lastUpdated, setLastUpdated] = useState(null);
  const autoRefreshRef = useRef(null);

  // Feature 6: City-Level Summary
  const [citySummary, setCitySummary] = useState(null);

  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const geoJsonLayerRef = useRef(null);
  const cityMarkersRef = useRef([]);
  const intervalRef = useRef(null);

  // Fetch traffic data & model metrics
  const fetchTraffic = useCallback(async (src, timeIdx, horizonVal, modelVal) => {
    try {
      const data = await api.getCurrentTraffic(src, timeIdx, horizonVal, modelVal);
      setTrafficData(data);
      setError(null);
      setLoading(false);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTraffic(datasetSource, demoTimeIndex, horizon, selectedModel);
    api.getModelPerformance().then(perf => setModelPerf(perf)).catch(() => {});
  }, [datasetSource, demoTimeIndex, horizon, selectedModel, fetchTraffic]);

  // Feature 6: Fetch city summary when india-mp is selected
  useEffect(() => {
    if (datasetSource === 'india-mp') {
      api.getCitySummary(demoTimeIndex).then(data => setCitySummary(data)).catch(() => {});
    } else {
      setCitySummary(null);
    }
  }, [datasetSource, demoTimeIndex]);

  // Feature 5: Auto-refresh every 60 seconds
  useEffect(() => {
    autoRefreshRef.current = setInterval(() => {
      fetchTraffic(datasetSource, demoTimeIndex, horizon, selectedModel);
      if (datasetSource === 'india-mp') {
        api.getCitySummary(demoTimeIndex).then(data => setCitySummary(data)).catch(() => {});
      }
    }, 60000);
    return () => clearInterval(autoRefreshRef.current);
  }, [datasetSource, demoTimeIndex, horizon, selectedModel, fetchTraffic]);

  const handleManualRefresh = () => {
    fetchTraffic(datasetSource, demoTimeIndex, horizon, selectedModel);
    if (datasetSource === 'india-mp') {
      api.getCitySummary(demoTimeIndex).then(data => setCitySummary(data)).catch(() => {});
    }
  };

  const handleDatasetChange = (newSource) => {
    setDatasetSource(newSource);
    setSelectedSensor(null);
    setForecast(null);
    setCityNotice(null);
    handleResetSimulation();
    const newViewport = REGION_VIEWPORTS[newSource] || REGION_VIEWPORTS['india-mp'];
    setActiveArea(newViewport);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([newViewport.lat, newViewport.lon], newViewport.zoom, { duration: 1.2 });
      setTimeout(() => { if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize(); }, 300);
    }
  };

  // Demo playback loop
  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setDemoTimeIndex(prev => {
          const next = prev + 1;
          if (trafficData && next >= trafficData.total_timestamps) {
            setPlaying(false);
            return prev;
          }
          return next;
        });
      }, 1500);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, trafficData]);

  // Initialize Map ONLY AFTER DOM container is rendered (loading === false)
  useEffect(() => {
    if (loading || !mapRef.current || mapInstanceRef.current) return;

    const initialViewport = REGION_VIEWPORTS['india-mp'];

    const map = L.map(mapRef.current, {
      center: [initialViewport.lat, initialViewport.lon],
      zoom: initialViewport.zoom,
      zoomControl: true,
      attributionControl: false,
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', {
      maxZoom: 19,
      subdomains: 'abcd',
    }).addTo(map);

    mapInstanceRef.current = map;

    // Add Madhya Pradesh State Boundary Polygon
    if (datasetSource === 'india-mp' && mpBoundaryData) {
      geoJsonLayerRef.current = L.geoJSON(mpBoundaryData, {
        style: {
          color: '#00d4ff',
          weight: 2,
          opacity: 0.85,
          fillColor: '#00d4ff',
          fillOpacity: 0.04,
          dashArray: '4,4',
        },
      }).addTo(map);
    }

    const timer = setTimeout(() => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [loading]);

  // Update MP State Boundary when dataset source changes
  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    if (geoJsonLayerRef.current) {
      geoJsonLayerRef.current.remove();
      geoJsonLayerRef.current = null;
    }

    if (datasetSource === 'india-mp' && mpBoundaryData) {
      geoJsonLayerRef.current = L.geoJSON(mpBoundaryData, {
        style: {
          color: '#00d4ff',
          weight: 2,
          opacity: 0.85,
          fillColor: '#00d4ff',
          fillOpacity: 0.04,
          dashArray: '4,4',
        },
      }).addTo(map);
    }
  }, [datasetSource]);

  // Handle map flyTo area / city selection
  const handleSelectArea = (area) => {
    setActiveArea(area);
    setCityNotice(null);

    if (area.hasData === false) {
      setCityNotice(`City Reference: No traffic sensor data currently available for ${area.name}. Demonstration sensor coverage is active in Bhopal, Indore, Ujjain, Gwalior, and Jabalpur.`);
    }

    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([area.lat, area.lon], area.zoom, { duration: 1.2 });
      setTimeout(() => { if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize(); }, 400);
    }
  };

  // Reset Map View to Entire MP State
  const handleResetToState = () => {
    const stateViewport = REGION_VIEWPORTS['india-mp'];
    setActiveArea(stateViewport);
    setCityNotice(null);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([stateViewport.lat, stateViewport.lon], stateViewport.zoom, { duration: 1.2 });
      setTimeout(() => { if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize(); }, 400);
    }
  };

  // Feature 2: Disruption simulation map overlay
  const clearSimMarkers = () => {
    simMarkersRef.current.forEach(m => m.remove());
    simMarkersRef.current = [];
  };

  useEffect(() => {
    if (!mapInstanceRef.current || !simResult || !simResult.affected_sensors) {
      clearSimMarkers();
      return;
    }
    const map = mapInstanceRef.current;
    clearSimMarkers();

    // Source sensor: pulsing red
    const src = simResult.source_sensor;
    if (src) {
      const srcMarker = L.circleMarker([src.latitude, src.longitude], {
        radius: 14,
        fillColor: '#ef4444',
        color: '#ffffff',
        weight: 3,
        opacity: 1,
        fillOpacity: 0.9,
        className: 'sim-pulse-marker',
      }).addTo(map);
      srcMarker.bindPopup(`<div style="font-family:Inter,sans-serif;padding:4px"><strong style="color:#ef4444">⚠ SIMULATED INCIDENT</strong><br/>${src.road_name}<br/>Speed: ${src.original_speed} → ${src.simulated_speed} mph</div>`);
      simMarkersRef.current.push(srcMarker);
    }

    // Affected sensors: orange with varying opacity
    simResult.affected_sensors.forEach(sensor => {
      const intensity = Math.min(1, sensor.speed_reduction_pct / 50);
      const color = intensity > 0.5 ? '#ef4444' : '#f59e0b';
      const marker = L.circleMarker([sensor.latitude, sensor.longitude], {
        radius: 9 + intensity * 4,
        fillColor: color,
        color: color,
        weight: 2,
        opacity: 0.9,
        fillOpacity: 0.4 + intensity * 0.4,
      }).addTo(map);
      marker.bindPopup(`<div style="font-family:Inter,sans-serif;padding:4px"><strong>${sensor.road_name}</strong><br/>Speed: ${sensor.original_speed} → ${sensor.reduced_speed} mph<br/><span style="color:${color}">-${sensor.speed_reduction_pct}% reduction</span></div>`);
      simMarkersRef.current.push(marker);
    });

    return () => clearSimMarkers();
  }, [simResult]);

  // Render Map Sensor Markers & Reference City Markers
  useEffect(() => {
    if (!mapInstanceRef.current || !trafficData?.sensors) return;
    const map = mapInstanceRef.current;

    // Clear old sensor markers
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    // Clear old city reference markers
    cityMarkersRef.current.forEach(c => c.remove());
    cityMarkersRef.current = [];

    const colorMap = {
      'Free Flow': '#10b981',
      'Moderate': '#f59e0b',
      'Congested': '#ef4444',
    };

    // Render MP City Reference Labels for geographic orientation
    if (datasetSource === 'india-mp') {
      MP_REFERENCE_CITIES.forEach(city => {
        const cityIcon = L.divIcon({
          className: 'city-reference-marker',
          html: `<div class="city-pin-badge ${city.hasData ? 'has-data' : ''}">${city.icon} ${city.name.split(' ')[0]}</div>`,
          iconSize: [80, 24],
          iconAnchor: [40, 12],
        });

        const cMarker = L.marker([city.lat, city.lon], { icon: cityIcon }).addTo(map);
        cMarker.on('click', () => handleSelectArea(city));
        cityMarkersRef.current.push(cMarker);
      });
    }

    // Filter sensors
    const filteredSensors = trafficData.sensors.filter(s => {
      if (searchQuery !== '') {
        const q = searchQuery.toLowerCase();
        return (
          s.sensor_id.toLowerCase().includes(q) ||
          s.congestion_state.toLowerCase().includes(q) ||
          (s.city && s.city.toLowerCase().includes(q)) ||
          (s.road_name && s.road_name.toLowerCase().includes(q))
        );
      }
      return true;
    });

    filteredSensors.forEach(sensor => {
      const color = colorMap[sensor.congestion_state] || '#64748b';
      const isSelected = selectedSensor === sensor.sensor_id;

      const marker = L.circleMarker([sensor.latitude, sensor.longitude], {
        radius: isSelected ? 10 : 7,
        fillColor: color,
        color: isSelected ? '#ffffff' : color,
        weight: isSelected ? 3 : 1,
        opacity: 0.95,
        fillOpacity: 0.85,
      }).addTo(map);

      const titleText = horizon === 0 ? 'Current Speed' : `Predicted Speed (${horizon}m)`;
      const displayName = sensor.road_name || sensor.sensor_id;
      const cityText = sensor.city ? ` (${sensor.city})` : '';

      marker.bindPopup(`
        <div style="font-family:Inter,sans-serif;min-width:180px;padding:4px">
          <strong style="color:#0a0e1a;font-size:14px">${displayName}${cityText}</strong><br/>
          <span style="color:#64748b;font-size:11px">ID: ${sensor.sensor_id}</span><br/>
          <span style="color:#475569;font-size:12px">${titleText}: <b>${sensor.speed} mph</b></span><br/>
          <span style="color:${color};font-weight:700;font-size:12px">${sensor.congestion_state}</span>
        </div>
      `);

      marker.on('click', () => handleSensorSelect(sensor.sensor_id, sensor.latitude, sensor.longitude));
      markersRef.current.push(marker);
    });

    map.invalidateSize();
  }, [trafficData, searchQuery, selectedSensor, horizon, datasetSource]);

  // Handle Sensor Selection
  const handleSensorSelect = async (sensorId, lat, lon) => {
    if (!sensorId) return;
    setSelectedSensor(sensorId);
    setCityNotice(null);
    setMultiModelData(null);
    setFeatureImportance(null);
    setShowFeatureImportance(false);

    if (lat && lon && mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([lat, lon], 14, { duration: 1.0 });
    } else if (trafficData?.sensors) {
      const found = trafficData.sensors.find(s => s.sensor_id === sensorId);
      if (found && mapInstanceRef.current) {
        mapInstanceRef.current.flyTo([found.latitude, found.longitude], 14, { duration: 1.0 });
      }
    }

    try {
      const data = await api.getForecast(sensorId, selectedModel, demoTimeIndex);
      setForecast(data);
    } catch (e) {
      console.error('Forecast error:', e);
    }

    // Feature 3: Fetch feature importance
    api.getFeatureImportance(sensorId, selectedModel, demoTimeIndex)
      .then(data => setFeatureImportance(data))
      .catch(() => {});
  };

  // Feature 1: Fetch multi-model data when comparison toggles change
  useEffect(() => {
    const activeCount = Object.values(compareModels).filter(Boolean).length;
    if (activeCount > 1 && selectedSensor) {
      api.getMultiModelForecast(selectedSensor, demoTimeIndex)
        .then(data => setMultiModelData(data))
        .catch(() => setMultiModelData(null));
    } else {
      setMultiModelData(null);
    }
  }, [compareModels, selectedSensor, demoTimeIndex]);

  // Feature 2: Disruption simulation handlers
  const handleSimulate = async () => {
    if (!selectedSensor) return;
    setSimLoading(true);
    try {
      const result = await api.simulateDisruption(selectedSensor, simDuration, datasetSource, demoTimeIndex);
      setSimResult(result);
    } catch (e) {
      console.error('Simulation error:', e);
    }
    setSimLoading(false);
  };

  const handleResetSimulation = () => {
    setSimResult(null);
    clearSimMarkers();
    api.resetSimulation().catch(() => {});
  };

  const summary = trafficData?.summary || { free_flow: 0, moderate: 0, congested: 0, total: 0 };
  const horizonLabel = horizon === 0 ? 'CURRENT' : `${horizon}-MIN FORECAST`;
  const allSensorsList = trafficData?.sensors || [];
  const stgcnMetrics = modelPerf?.models?.find(m => m.model_name === 'stgcn');

  // Multi-model chart rendering helper
  const renderComparisonChart = () => {
    const activeModels = Object.entries(compareModels).filter(([, v]) => v).map(([k]) => k);
    const isMultiModel = activeModels.length > 1 && multiModelData?.models;

    // Determine data source
    let chartDataSets = {};
    if (isMultiModel) {
      activeModels.forEach(m => {
        if (multiModelData.models[m]) {
          chartDataSets[m] = multiModelData.models[m].actual_vs_predicted;
        }
      });
    } else if (forecast?.actual_vs_predicted) {
      chartDataSets[selectedModel] = forecast.actual_vs_predicted;
    }

    if (Object.keys(chartDataSets).length === 0) {
      return <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No trend data</p>;
    }

    // Get all values for scale
    const allVals = [];
    Object.values(chartDataSets).forEach(data => {
      data.forEach(d => {
        if (d.actual != null) allVals.push(d.actual);
        if (d.predicted != null) allVals.push(d.predicted);
        if (d.predicted_upper != null) allVals.push(d.predicted_upper);
        if (d.predicted_lower != null) allVals.push(d.predicted_lower);
      });
    });

    if (allVals.length === 0) return <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No trend data</p>;

    const minV = Math.min(...allVals) - 5;
    const maxV = Math.max(...allVals) + 5;
    const range = maxV - minV || 1;

    // Use the first dataset for actual line and x-axis
    const primaryData = Object.values(chartDataSets)[0];
    const xStep = 400 / Math.max(primaryData.length - 1, 1);
    const toY = (v) => 150 - ((v - minV) / range) * 140;

    // Actual line (same for all models — use primary)
    const actualLine = primaryData.map((d, i) => d.actual != null ? `${i * xStep},${toY(d.actual)}` : null).filter(Boolean).join(' ');

    return (
      <svg viewBox="0 0 400 150" className="speed-chart-svg">
        {/* Confidence band for each model */}
        {Object.entries(chartDataSets).map(([modelKey, data]) => {
          const bandPoints = [];
          const upperPoints = [];
          const lowerPoints = [];
          data.forEach((d, i) => {
            if (d.predicted_upper != null && d.predicted_lower != null) {
              upperPoints.push({ x: i * xStep, y: toY(d.predicted_upper) });
              lowerPoints.push({ x: i * xStep, y: toY(d.predicted_lower) });
            }
          });
          if (upperPoints.length > 1) {
            const bandPath = upperPoints.map(p => `${p.x},${p.y}`).join(' ') + ' ' +
              lowerPoints.reverse().map(p => `${p.x},${p.y}`).join(' ');
            const bandColor = MODEL_COLORS[modelKey]?.color || '#f59e0b';
            return <polygon key={`band-${modelKey}`} points={bandPath} fill={bandColor} fillOpacity="0.1" stroke="none" />;
          }
          return null;
        })}

        {/* Actual speed line */}
        <polyline points={actualLine} fill="none" stroke="#00d4ff" strokeWidth="2.5" opacity="0.9"/>

        {/* Predicted lines for each active model */}
        {Object.entries(chartDataSets).map(([modelKey, data]) => {
          const mc = MODEL_COLORS[modelKey] || { color: '#f59e0b', dash: '' };
          const predLine = data.map((d, i) => d.predicted != null ? `${i * xStep},${toY(d.predicted)}` : null).filter(Boolean).join(' ');
          if (!predLine) return null;
          return (
            <polyline
              key={`pred-${modelKey}`}
              points={predLine}
              fill="none"
              stroke={isMultiModel ? mc.color : '#f59e0b'}
              strokeWidth="2"
              strokeDasharray={isMultiModel ? mc.dash : '6,4'}
              opacity="0.85"
            />
          );
        })}
      </svg>
    );
  };

  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="page-header">
          <h1>Smart Traffic Congestion Forecasting</h1>
          <p>AI-powered prediction of urban traffic conditions using spatio-temporal machine learning.</p>
        </div>
        <div className="loading-spinner"></div>
        <p className="loading-text">Loading Madhya Pradesh traffic network dataset...</p>
      </motion.div>
    );
  }

  return (
    <div className="dashboard">
      {/* Hero Section */}
      <div className="page-header hero-section card">
        <div className="hero-content">
          <h1>Madhya Pradesh Urban Traffic Forecasting</h1>
          <p className="hero-subtitle">
            State-level traffic visualization with spatio-temporal ML predictions (STGCN, LSTM, XGBoost).
          </p>
        </div>
        
        {/* Compact System Status Box */}
        <div className="hero-status-box">
          <div className="status-item">
            <span className="status-item-label">SYSTEM STATUS</span>
            <span className="status-item-val green">● Operational</span>
          </div>
          <div className="status-divider"></div>
          <div className="status-item">
            <span className="status-item-label">MODEL</span>
            <span className="status-item-val accent">STGCN — Primary</span>
          </div>
          <div className="status-divider"></div>
          <div className="status-item">
            <span className="status-item-label">MODE</span>
            <span className="status-item-val purple">{demoMode ? 'Dataset Mode' : 'Live Data Mode'}</span>
          </div>
          <div className="status-divider"></div>
          <div className="status-item">
            <span className="status-item-label">FORECAST</span>
            <span className="status-item-val">30 / 60 min</span>
          </div>
          {/* Feature 5: Last Updated */}
          <div className="status-divider"></div>
          <div className="status-item">
            <span className="status-item-label">LAST UPDATED</span>
            <span className="status-item-val" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {lastUpdated ? lastUpdated.toLocaleTimeString() : '—'}
              <button
                className="refresh-btn"
                onClick={handleManualRefresh}
                title="Refresh data now"
              >
                🔄
              </button>
            </span>
          </div>
        </div>
      </div>

      {demoMode && (
        <div className="demo-banner">
          <strong>⚡ DEMO MODE</strong>
          <span>
            {datasetSource === 'india-mp'
              ? 'State of Madhya Pradesh: Sensor coverage active in demonstration corridors (Bhopal, Indore, Ujjain, Gwalior, Jabalpur).'
              : 'Simulated Traffic Network dataset.'}
          </span>
          <span className="badge badge-demo">Dataset Mode</span>
        </div>
      )}

      {cityNotice && (
        <div className="demo-banner" style={{ borderColor: 'var(--moderate)', color: 'var(--moderate)' }}>
          <strong>ℹ Notice:</strong> <span>{cityNotice}</span>
        </div>
      )}

      {error && (
        <div className="demo-banner" style={{ borderColor: 'var(--congested)', color: 'var(--congested)' }}>
          <strong>⚠ Error:</strong> <span>{error}</span>
        </div>
      )}

      {/* Feature 6: City-Level Summary Cards */}
      {datasetSource === 'india-mp' && citySummary?.cities && citySummary.cities.length > 0 && (
        <motion.div
          className="city-summary-grid"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.28, ease: [0.4,0,0.2,1] }}
        >
          {citySummary.cities.map((city, idx) => {
            const stateClass = city.congestion_state === 'Free Flow' ? 'free-flow' : city.congestion_state === 'Moderate' ? 'moderate' : 'congested';
            return (
              <motion.div
                key={city.city}
                className={`city-summary-card card ${stateClass}`}
                onClick={() => {
                  const ref = MP_REFERENCE_CITIES.find(c => c.name.startsWith(city.city));
                  if (ref) handleSelectArea(ref);
                }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: idx * 0.05, ease: [0.4,0,0.2,1] }}
                whileHover={{ y: -3, transition: { duration: 0.15 } }}
              >
                <div className="city-summary-header">
                  <span className="city-summary-icon">{city.icon}</span>
                  <span className="city-summary-name">{city.city}</span>
                </div>
                <div className="city-summary-speed">{city.avg_speed} mph</div>
                <span className={`badge badge-${stateClass === 'free-flow' ? 'success' : stateClass === 'moderate' ? 'warning' : 'danger'}`}>
                  {city.congestion_state}
                </span>
                <div className="city-summary-meta">
                  {city.sensor_count} sensors • {city.congested_pct}% congested
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      )}

      {/* KPI Cards Row */}
      <div className="status-grid">
        {[
          { cls: 'accent',    val: summary.total,      label: 'Active Sensors',  delay: 0    },
          { cls: 'congested', val: summary.congested,  label: horizon === 0 ? 'Congested Segments' : 'Predicted Congested', delay: 0.05 },
          { cls: 'moderate',  val: summary.moderate,   label: horizon === 0 ? 'Moderate Segments' : 'Predicted Moderate',  delay: 0.1  },
          { cls: 'free-flow', val: summary.free_flow,  label: horizon === 0 ? 'Free Flow Segments' : 'Predicted Free Flow', delay: 0.15 },
        ].map(({ cls, val, label, delay }) => (
          <motion.div
            key={label}
            className={`status-card ${cls} card`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.28, delay, ease: [0.4,0,0.2,1] }}
            whileHover={{ y: -2, transition: { duration: 0.12 } }}
          >
            <div className="status-value">{val}</div>
            <div className="status-label">{label}</div>
          </motion.div>
        ))}
        <motion.div
          className="status-card card"
          style={{ borderColor: 'rgba(0, 212, 255, 0.22)' }}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.28, delay: 0.2, ease: [0.4,0,0.2,1] }}
          whileHover={{ y: -2, transition: { duration: 0.12 } }}
        >
          <div className="status-value" style={{ fontSize: '1.55rem', color: 'var(--accent-primary)', fontWeight: 800, fontFamily: 'var(--font-mono)' }}>
            {stgcnMetrics?.mae != null ? stgcnMetrics.mae.toFixed(3) : '—'}
          </div>
          <div className="status-label">STGCN MAE Metric</div>
        </motion.div>
      </div>

      {/* Controls Bar */}
      <div className="dashboard-controls card" style={{ padding: '16px 20px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center', width: '100%' }}>
          
          {/* Region Selector */}
          <div className="control-group">
            <label>REGION / DATASET</label>
            <select
              className="form-select"
              value={datasetSource}
              onChange={(e) => handleDatasetChange(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: '8px', minWidth: '190px', fontWeight: 600, color: 'var(--accent-primary)' }}
            >
              <option value="india-mp">🇮🇳 India (Madhya Pradesh)</option>
              <option value="metr-la">🇺🇸 METR-LA (Los Angeles)</option>
              <option value="pems-bay">🇺🇸 PeMS-BAY (Bay Area)</option>
            </select>
          </div>

          {/* Famous Road Segment Selector */}
          <div className="control-group" style={{ flex: 1, minWidth: '260px' }}>
            <label>SELECT FAMOUS ROAD SEGMENT</label>
            <select
              className="form-select"
              value={selectedSensor || ''}
              onChange={(e) => handleSensorSelect(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: '8px', fontWeight: 600, color: 'var(--text-primary)' }}
            >
              <option value="">-- Choose Road (e.g. MG Road, VIP Road, Mahakal Corridor) --</option>
              {allSensorsList.map(s => (
                <option key={s.sensor_id} value={s.sensor_id}>
                  {s.road_name || s.name || s.sensor_id} {s.city ? `(${s.city})` : ''} — {s.speed} mph ({s.congestion_state})
                </option>
              ))}
            </select>
          </div>

          {/* Time Horizon Selector */}
          <div className="control-group">
            <label>TIME HORIZON ({horizonLabel})</label>
            <div className="horizon-pill-group">
              <button
                className={`horizon-pill ${horizon === 0 ? 'active' : ''}`}
                onClick={() => setHorizon(0)}
              >
                LIVE (NOW)
              </button>
              <button
                className={`horizon-pill ${horizon === 30 ? 'active' : ''}`}
                onClick={() => setHorizon(30)}
              >
                🔮 30 MIN FORECAST
              </button>
              <button
                className={`horizon-pill ${horizon === 60 ? 'active' : ''}`}
                onClick={() => setHorizon(60)}
              >
                🔮 60 MIN FORECAST
              </button>
            </div>
          </div>

          {/* Model Selector */}
          <div className="control-group">
            <label>MODEL</label>
            <select
              className="form-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: '8px', minWidth: '120px' }}
            >
              <option value="stgcn">STGCN (Primary)</option>
              <option value="lstm">LSTM (Secondary)</option>
              <option value="xgboost">XGBoost (Baseline)</option>
            </select>
          </div>

          {/* Time Playback */}
          <div className="control-group demo-controls">
            <label>DEMO TIME: {trafficData?.timestamp?.slice(0, 16) || '—'}</label>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <button className="btn btn-sm btn-secondary" onClick={() => setDemoTimeIndex(Math.max(0, demoTimeIndex - 10))}>◀◀</button>
              <button className="btn btn-sm btn-secondary" onClick={() => setDemoTimeIndex(Math.max(0, demoTimeIndex - 1))}>◀</button>
              <button className="btn btn-sm btn-primary" onClick={() => setPlaying(!playing)}>{playing ? '⏸' : '▶'}</button>
              <button className="btn btn-sm btn-secondary" onClick={() => setDemoTimeIndex(demoTimeIndex + 1)}>▶</button>
              <button className="btn btn-sm btn-secondary" onClick={() => setDemoTimeIndex(demoTimeIndex + 10)}>▶▶</button>
            </div>
          </div>

        </div>

        {/* Quick Area Jump & Reset View Controls */}
        <div className="area-quick-filters">
          {datasetSource === 'india-mp' && (
            <button className="btn btn-sm btn-primary" onClick={handleResetToState} style={{ padding: '4px 12px', fontSize: '0.75rem' }}>
              🗺️ Entire MP State View
            </button>
          )}

          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginLeft: 6 }}>MP CITIES & REFERENCE LOCATIONS:</span>
          {MP_REFERENCE_CITIES.map(city => (
            <button
              key={city.name}
              className={`area-btn ${activeArea.name === city.name ? 'active' : ''}`}
              onClick={() => handleSelectArea(city)}
            >
              <span>{city.icon}</span> {city.name.split(' ')[0]}
              {!city.hasData && <span style={{ fontSize: '0.65rem', opacity: 0.6, marginLeft: 2 }}>(Ref)</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid (Map + Detail Panel) */}
      <div className="dashboard-main">
        {/* Interactive Map Section */}
        <div className="map-section card">
          <div className="card-header">
            <div>
              <span className="card-title">
                📍 Map — {datasetSource === 'india-mp' ? 'State of Madhya Pradesh, India' : datasetSource.toUpperCase()} : {activeArea.name}
              </span>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {datasetSource === 'india-mp'
                  ? 'Sensor coverage: Demonstration dataset (Bhopal, Indore, Ujjain, Gwalior, Jabalpur)'
                  : 'Traffic sensor network map'}
              </div>
            </div>
            <span className="badge badge-primary">{markersRef.current.length} visible nodes</span>
          </div>

          <div className="map-container" ref={mapRef} style={{ height: '540px' }}>
            {datasetSource === 'india-mp' && (
              <div className={`map-coverage-box ${showCoverageOverlay ? 'expanded' : 'collapsed'}`}>
                <button
                  className="coverage-toggle-btn"
                  onClick={(e) => { e.stopPropagation(); setShowCoverageOverlay(!showCoverageOverlay); }}
                  title="Toggle MP Coverage Info"
                >
                  <span>ℹ MP Coverage</span>
                  <span className="toggle-icon">{showCoverageOverlay ? '▲' : '▼'}</span>
                </button>

                <AnimatePresence initial={false}>
                  {showCoverageOverlay && (
                    <motion.div
                      className="coverage-details"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2, ease: [0.4,0,0.2,1] }}
                      style={{ overflow: 'hidden' }}
                    >
                      <div className="coverage-row">
                        <span className="coverage-key">Geographic Coverage:</span>
                        <span className="coverage-value cyan">Madhya Pradesh — 100%</span>
                      </div>
                      <div className="coverage-row">
                        <span className="coverage-key">Active Sensor Coverage:</span>
                        <span className="coverage-value accent">5 cities</span>
                      </div>
                      <div className="coverage-row">
                        <span className="coverage-key">Traffic Forecasting:</span>
                        <span className="coverage-value green">Available at active sensors</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>

          <div className="map-legend">
            <div className="legend-item"><span className="legend-dot" style={{ background: 'var(--free-flow)' }}></span> 🟢 FREE FLOW (&ge;45 mph)</div>
            <div className="legend-item"><span className="legend-dot" style={{ background: 'var(--moderate)' }}></span> 🟡 MODERATE (25-45 mph)</div>
            <div className="legend-item"><span className="legend-dot" style={{ background: 'var(--congested)' }}></span> 🔴 CONGESTED (&lt;25 mph)</div>
            <div className="legend-item"><span className="legend-dot-boundary"></span> ▱ MP State Boundary</div>
            <div className="legend-item"><span className="legend-dot-city"></span> 📍 City Reference</div>
          </div>
        </div>

        {/* Sensor Detail & Forecast Side Panel */}
        <motion.div
          className={`sensor-detail-panel card ${selectedSensor ? 'visible' : ''}`}
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.25, ease: [0.4,0,0.2,1] }}
        >
          {forecast ? (
            <>
              <div className="card-header">
                <div>
                  <span className="card-title" style={{ fontSize: '1.2rem' }}>
                    {forecast.road_name || forecast.name || forecast.sensor_id}
                  </span>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {forecast.sensor_id}</div>
                  {forecast.city && <div style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', fontWeight: 600, marginTop: 2 }}>Location: {forecast.city}, Madhya Pradesh</div>}
                </div>
                <button className="btn btn-ghost btn-sm" onClick={() => { setSelectedSensor(null); setForecast(null); setSimResult(null); clearSimMarkers(); }}>✕</button>
              </div>

              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Current Speed</span>
                  <span className="detail-value">{forecast.current_speed} mph</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Current Status</span>
                  <span className={`badge badge-${forecast.current_congestion === 'Free Flow' ? 'success' : forecast.current_congestion === 'Moderate' ? 'warning' : 'danger'}`}>
                    {forecast.current_congestion}
                  </span>
                </div>
              </div>

              <h4 style={{ margin: '16px 0 8px', color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Short-Term Traffic Forecasts ({selectedModel.toUpperCase()})
              </h4>

              <div className="forecast-cards">
                <div className={`forecast-card ${horizon === 30 ? 'highlight' : ''}`}>
                  <span className="forecast-horizon">🔮 30 MIN FORECAST</span>
                  <span className="forecast-speed">{forecast.forecasts['30_min'].predicted_speed} mph</span>
                  <span className={`badge badge-${forecast.forecasts['30_min'].congestion_state === 'Free Flow' ? 'success' : forecast.forecasts['30_min'].congestion_state === 'Moderate' ? 'warning' : 'danger'}`}>
                    {forecast.forecasts['30_min'].congestion_state}
                  </span>
                </div>
                <div className={`forecast-card ${horizon === 60 ? 'highlight' : ''}`}>
                  <span className="forecast-horizon">🔮 60 MIN FORECAST</span>
                  <span className="forecast-speed">{forecast.forecasts['60_min'].predicted_speed} mph</span>
                  <span className={`badge badge-${forecast.forecasts['60_min'].congestion_state === 'Free Flow' ? 'success' : forecast.forecasts['60_min'].congestion_state === 'Moderate' ? 'warning' : 'danger'}`}>
                    {forecast.forecasts['60_min'].congestion_state}
                  </span>
                </div>
              </div>

              {/* Predicted vs Actual Speed Section */}
              <div style={{ marginTop: 20 }}>
                <h4 style={{ margin: '0 0 2px', color: 'var(--text-primary)', fontSize: '0.9rem', fontWeight: 700 }}>
                  Predicted vs Actual Traffic Speed
                </h4>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 8 }}>
                  {Object.values(compareModels).filter(Boolean).length > 1
                    ? 'Multi-model comparison view'
                    : `${selectedModel.toUpperCase()} forecast performance for selected road segment`}
                </div>

                {/* Feature 1: Model Comparison Toggle */}
                <div className="model-compare-group">
                  {Object.entries(MODEL_COLORS).map(([key, mc]) => (
                    <label key={key} className="model-compare-checkbox">
                      <input
                        type="checkbox"
                        checked={compareModels[key]}
                        onChange={(e) => setCompareModels(prev => ({ ...prev, [key]: e.target.checked }))}
                      />
                      <span className="model-compare-dot" style={{ background: mc.color }}></span>
                      {mc.label}
                    </label>
                  ))}
                </div>

                <div className="mini-chart">
                  {renderComparisonChart()}
                  
                  <div className="chart-legend-inline">
                    <span><span style={{ color: '#00d4ff', fontWeight: 'bold' }}>━</span> Actual Speed</span>
                    {Object.values(compareModels).filter(Boolean).length > 1 ? (
                      Object.entries(compareModels).filter(([, v]) => v).map(([key]) => (
                        <span key={key}>
                          <span style={{ color: MODEL_COLORS[key].color, fontWeight: 'bold' }}>╌</span> {MODEL_COLORS[key].label}
                        </span>
                      ))
                    ) : (
                      <span><span style={{ color: '#f59e0b', fontWeight: 'bold' }}>╌</span> Predicted Speed</span>
                    )}
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>░ Confidence Range</span>
                  </div>
                </div>
              </div>

              {/* Feature 2: Disruption Simulator */}
              <div className="disruption-section">
                <h4 style={{ margin: '16px 0 8px', color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  ⚡ What-If Disruption Simulator
                </h4>
                {!simResult ? (
                  <div className="disruption-controls">
                    <select
                      className="form-select"
                      value={simDuration}
                      onChange={(e) => setSimDuration(Number(e.target.value))}
                      style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', flex: '0 0 auto' }}
                    >
                      <option value={30}>30 min propagation</option>
                      <option value={60}>60 min propagation</option>
                    </select>
                    <button
                      className="btn btn-sm disruption-btn"
                      onClick={handleSimulate}
                      disabled={simLoading}
                    >
                      {simLoading ? '⏳ Simulating...' : '⚠ Simulate Incident Here'}
                    </button>
                  </div>
                ) : (
                  <div className="disruption-result">
                    <div className="disruption-summary">
                      <span className="disruption-count">{simResult.total_affected}</span>
                      <span>neighboring sensors affected ({simResult.duration_minutes} min propagation)</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 8 }}>
                      {simResult.propagation_note}
                    </div>
                    {simResult.affected_sensors.slice(0, 5).map(s => (
                      <div key={s.sensor_id} className="disruption-affected-row">
                        <span>{s.road_name}</span>
                        <span style={{ color: s.projected_state === 'Congested' ? 'var(--congested)' : 'var(--moderate)' }}>
                          {s.original_speed} → {s.reduced_speed} mph (−{s.speed_reduction_pct}%)
                        </span>
                      </div>
                    ))}
                    <button className="btn btn-sm btn-secondary" onClick={handleResetSimulation} style={{ marginTop: 8 }}>
                      ↩ Reset Simulation
                    </button>
                  </div>
                )}
              </div>

              {/* Feature 3: Feature Importance Panel */}
              <div className="fi-section">
                <button
                  className="fi-toggle-btn"
                  onClick={() => setShowFeatureImportance(!showFeatureImportance)}
                >
                  <span>📊 Feature Importance ({selectedModel.toUpperCase()})</span>
                  <span className="toggle-icon">{showFeatureImportance ? '▲' : '▼'}</span>
                </button>
                {showFeatureImportance && featureImportance?.features && (
                  <div className="fi-bar-chart">
                    {featureImportance.features.slice(0, 6).map((f, idx) => (
                      <div key={idx} className="fi-bar-row">
                        <span className="fi-bar-label">{f.feature}</span>
                        <div className="fi-bar-track">
                          <div
                            className="fi-bar-fill"
                            style={{ width: `${f.importance_pct}%` }}
                          ></div>
                        </div>
                        <span className="fi-bar-value">{f.importance_pct}%</span>
                      </div>
                    ))}
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 6 }}>
                      Source: {featureImportance.source}
                    </div>
                  </div>
                )}
              </div>

              <div style={{ marginTop: 12, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Model: {selectedModel.toUpperCase()} • {forecast.forecast_source}
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
              <p style={{ fontSize: '2.5rem', marginBottom: 12 }}>🗺️</p>
              <p style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Select a Road Segment</p>
              <p style={{ fontSize: '0.85rem', marginTop: 4 }}>
                Choose a road from the dropdown above or click any node in Madhya Pradesh to view 30-min & 60-min traffic forecasts.
              </p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
