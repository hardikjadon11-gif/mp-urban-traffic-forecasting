/**
 * API Service — Centralized backend communication layer.
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';

async function fetchApi(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    if (error.message.includes('fetch')) {
      throw new Error('Cannot connect to backend. Is the server running?');
    }
    throw error;
  }
}

export const api = {
  // Health
  health: () => fetchApi('/health'),

  // Datasets
  getDatasets: () => fetchApi('/datasets'),

  // Sensors
  getSensors: (source) => fetchApi(`/sensors${source ? `?source=${source}` : ''}`),
  getSensor: (id) => fetchApi(`/sensors/${id}`),

  // Traffic
  getCurrentTraffic: (source = 'metr-la', demoTimeIndex, horizon = 0, model = 'stgcn') => {
    let url = `/traffic/current?source=${source}&horizon=${horizon}&model=${model}`;
    if (demoTimeIndex !== undefined) url += `&demo_time_index=${demoTimeIndex}`;
    return fetchApi(url);
  },

  // City Summary (India MP)
  getCitySummary: (demoTimeIndex) => {
    let url = '/traffic/city-summary';
    if (demoTimeIndex !== undefined) url += `?demo_time_index=${demoTimeIndex}`;
    return fetchApi(url);
  },

  // Forecast
  getForecast: (sensorId, model = 'stgcn', demoTimeIndex) => {
    let url = `/forecast/${sensorId}?model=${model}`;
    if (demoTimeIndex !== undefined) url += `&demo_time_index=${demoTimeIndex}`;
    return fetchApi(url);
  },

  // Multi-Model Forecast Comparison
  getMultiModelForecast: (sensorId, demoTimeIndex) => {
    let url = `/forecast/compare/${sensorId}`;
    if (demoTimeIndex !== undefined) url += `?demo_time_index=${demoTimeIndex}`;
    return fetchApi(url);
  },

  // Feature Importance
  getFeatureImportance: (sensorId, model = 'stgcn', demoTimeIndex) => {
    let url = `/feature-importance/${sensorId}?model=${model}`;
    if (demoTimeIndex !== undefined) url += `&demo_time_index=${demoTimeIndex}`;
    return fetchApi(url);
  },

  // Disruption Simulation
  simulateDisruption: (sensorId, durationMinutes = 30, source = 'india-mp', demoTimeIndex) => {
    let url = `/simulate/disruption?sensor_id=${sensorId}&duration_minutes=${durationMinutes}&source=${source}`;
    if (demoTimeIndex !== undefined) url += `&demo_time_index=${demoTimeIndex}`;
    return fetchApi(url, { method: 'POST' });
  },
  resetSimulation: () => fetchApi('/simulate/reset', { method: 'POST' }),

  // Training
  startTraining: (config) => fetchApi('/train', {
    method: 'POST',
    body: JSON.stringify(config),
  }),
  getTrainingStatus: () => fetchApi('/train/status'),
  getModelTrainingStatus: (model) => fetchApi(`/train/status/${model}`),

  // Evaluation
  getModelPerformance: () => fetchApi('/model-performance'),
  triggerEvaluation: () => fetchApi('/evaluate', { method: 'POST' }),

  // Fusion
  getFusionStatus: () => fetchApi('/data-fusion/status'),
};

export default api;
