# 🛰️ VisionRadar — Monocular Traffic Intelligence & Speed Estimation Platform

> An Error-Decomposed, Uncertainty-Aware Monocular Vehicle Speed Estimation Platform & Real-Time Traffic Analytics Workbench.

![VisionRadar Architecture](https://img.shields.io/badge/VisionRadar-v0.1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌟 Overview

**VisionRadar** is an end-to-end Computer Vision & AI Traffic Analytics platform designed for monocular road surveillance cameras. It combines real-time deep learning detection (YOLOX-Nano ONNX), multi-object tracking (ByteTrack), camera homography calibration, and speed uncertainty decomposition into a unified real-time dashboard.

---

## ✨ Key Features & Capabilities

- 🚗 **Real-Time Vehicle Detection**: Powered by **YOLOX-Nano ONNX** executed via OpenCV-DNN / ONNXRuntime CPU execution providers.
- 🎯 **Multi-Object Tracking (MOT)**: **ByteTrack** algorithm maintaining persistent track IDs, trajectory history, and handling occlusions.
- 📐 **3D Homography Calibration**: Interactive 4-point ground-plane calibration mapping 2D video pixel coordinates to real-world metric distances (meters).
- ⚡ **Speed & Uncertainty Estimation**: Error-decomposed speed calculation (instantaneous & smoothed km/h) with error variance bounds and ROI validity checks.
- 🚨 **Automated Violation Detection & Evidence**: Real-time speed limit violation enforcement with automatic cropped/full-frame evidence capture.
- 📡 **Dual Delivery Architecture**:
  - **WebSocket Real-Time Stream**: Live frame-by-frame detection broadcasts (`ws://`).
  - **REST API & Polling Auto-Recovery**: Background worker fallback ensuring zero lost data if WebSockets disconnect.
- 🖥️ **Interactive Canvas Workbench**: Modern React SPA featuring dynamic HUD overlays, vehicle speed badges, trajectory tails, violation alerts, and telemetry inspection panels.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │             React + TypeScript SPA           │
                               │  (Interactive Canvas Workbench & Analytics) │
                               └──────────────────────┬───────────────────────┘
                                                      │ HTTP / WebSocket (ws://)
                               ┌──────────────────────▼───────────────────────┐
                               │           FastAPI Backend Engine             │
                               │      (REST Endpoints & WebSocket Server)     │
                               └──────────┬─────────────────────────┬─────────┘
                                          │                         │
               ┌──────────────────────────▼──────────┐   ┌──────────▼──────────────────────────┐
               │    Perception & Intelligence Worker │   │        SQLite Database (WAL)        │
               │  • YOLOX-Nano ONNX Detector         │   │  • Video & Job Telemetry            │
               │  • ByteTrack Multi-Object Tracker   │   │  • Trajectories & Speed Records     │
               │  • Homography Speed Estimator       │   │  • Violation Evidence Metadata      │
               └─────────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```
VisionRadar/
├── api/
│   └── index.py                   # Vercel Serverless Function entrypoint
├── apps/
│   ├── api/                       # FastAPI Server Application
│   │   ├── main.py                # Core REST & WebSocket Endpoints
│   │   ├── routes_stream.py       # Real-Time Perception Streaming Router
│   │   └── schemas.py             # Pydantic Schemas
│   └── web/                       # React 18 SPA Frontend
│       ├── src/
│       │   ├── components/        # WorkbenchView, Analytics, Header
│       │   ├── App.tsx            # Main App Layout & Tabs
│       │   └── index.css          # Design System & Tailwind Directives
│       ├── package.json
│       └── vite.config.ts         # Vite Config with WebSocket Proxy (ws: true)
├── packages/
│   └── visionradar/               # Core Python AI Engine Package
│       ├── cv/                    # Detector, Tracker, Calibration, Speed Estimator
│       ├── models/                # SQLAlchemy Models & DB Initialization
│       ├── perception/            # Perception Pipeline & Metrics Collector
│       └── worker/                # Real-Time Job Streaming Worker
├── scripts/                       # Database Seeding & Verification Scripts
├── tests/                         # Pytest Pipeline & API Test Suite
├── vercel.json                    # Deployment Configuration for Vercel
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quick Start & Local Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm 9+**

### 1. Clone the Repository
```bash
git clone https://github.com/GloriousKrrish/Vision-Radar.git
cd Vision-Radar
```

### 2. Set Up Python Backend
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Seed initial database records
python scripts/seed_database.py
```

### 3. Run FastAPI Backend Server
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```
*API documentation available at `http://localhost:8000/docs`.*

### 4. Run Frontend Workbench
In a separate terminal:
```bash
cd apps/web
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser.

---

## 🌐 Production Deployment (Vercel)

VisionRadar is configured for Vercel deployment via `vercel.json`:

```json
{
  "version": 2,
  "buildCommand": "cd apps/web && npm install && npm run build",
  "outputDirectory": "apps/web/dist",
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index.py" },
    { "source": "/((?!api/).*)", "destination": "/$1" }
  ]
}
```

### Deploying via Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to preview / production
vercel --prod
```

Or push directly to **`GitHub main`** branch with Vercel GitHub integration enabled.

---

## 🧪 Running Verification Tests

```bash
# Run backend pytest suite
pytest tests/ -v
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
