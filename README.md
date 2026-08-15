---
title: MP Urban Traffic Forecasting
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: AI-powered Urban Mobility & Traffic Congestion Forecasting System for Madhya Pradesh
---

# Smart Traffic Congestion Forecasting System

AI-powered urban mobility management system featuring Spatio-Temporal Graph Convolutional Networks (STGCN), LSTM, and XGBoost models for traffic congestion prediction across Madhya Pradesh and benchmark datasets (METR-LA, PeMS-BAY).

## Key Features
- **Madhya Pradesh Coverage**: Live sensor data and traffic forecasting across MP highway corridors (Bhopal, Indore, Gwalior, Jabalpur, Ujjain, Sagar, Rewa, Ratlam).
- **Multi-Model Suite**: STGCN, LSTM, XGBoost for 30 min & 60 min horizon forecasting.
- **Interactive Dashboard**: Leaflet map visualizer with congestion heatmaps and real-time sensor analytics.
- **Multi-Dataset Fusion**: Support for MP Highways, METR-LA, PeMS-BAY, and Uber Movement dataset fusion.

## Unified Container Setup
This repository uses the Unified Docker SDK combining the FastAPI REST backend and React Vite SPA into a single production container listening on port `7860`.
