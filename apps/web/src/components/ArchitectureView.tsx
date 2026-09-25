import React, { useState, useEffect } from 'react';

const EP: Record<string, { b: any; r: any }> = {
  'GET /api/v1/health': {
    b: null,
    r: { status: 'online', version: '0.1.0', cv_engine: 'active', database: 'sqlite_wal' }
  },
  'GET /api/v1/projects': {
    b: null,
    r: [{ id: 1, name: 'Highway Speed & Flow Study', created_at: '2026-09-24T12:00:00Z' }]
  },
  'POST /api/v1/projects': {
    b: { name: 'Junction Traffic Study', description: 'Monocular speed benchmark' },
    r: { id: 2, name: 'Junction Traffic Study', created_at: '2026-09-24T12:05:00Z' }
  },
  'POST /api/v1/videos/1/calibrations': {
    b: {
      image_points: [[330, 160], [470, 160], [748, 435], [51, 435]],
      world_points: [[-1, 150], [13, 150], [13, 0.5], [-1, 0.5]]
    },
    r: { id: 1, video_id: 1, version: 1, reprojection_rmse_m: 0.31, camera_height_m: 9.0, pitch_deg: 14.0 }
  },
  'GET /api/v1/jobs/1/tracks': {
    b: null,
    r: [
      { id: 1, track_id: 1, vehicle_class: 'Sedan', confidence: 0.94, first_frame: 0, last_frame: 300 }
    ]
  },
  'GET /api/v1/jobs/1/violations': {
    b: null,
    r: [
      {
        id: 1,
        track_id: 2,
        estimated_speed_kmh: 83.7,
        speed_limit_kmh: 80.0,
        review_status: 'PENDING'
      }
    ]
  },
  'GET /api/v1/experiments': {
    b: null,
    r: [
      { name: 'BrnoCompSpeed Benchmark', mae_kmh: 1.92, rmse_kmh: 2.45, r2_score: 0.9812 },
      { name: 'UA-DETRAC Benchmark', mae_kmh: 3.41, rmse_kmh: 4.12, r2_score: 0.945 }
    ]
  }
};

export const ArchitectureView: React.FC = () => {
  const [selectedEp, setSelectedEp] = useState<string>('GET /api/v1/health');
  const [liveResponse, setLiveResponse] = useState<any>(EP['GET /api/v1/health'].r);

  const [method, path] = selectedEp.split(' ');
  const epData = EP[selectedEp] || { b: null, r: {} };

  // Fetch live response from backend when selected
  useEffect(() => {
    if (method === 'GET') {
      fetch(path)
        .then(res => res.json())
        .then(data => setLiveResponse(data))
        .catch(() => setLiveResponse(epData.r));
    } else {
      setLiveResponse(epData.r);
    }
  }, [selectedEp]);

  const curlStr =
    method === 'WS'
      ? `wscat -c ws://localhost:8000${path}`
      : `curl -X ${method} http://localhost:8000${path}` +
        (epData.b ? ` \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(epData.b)}'` : '');

  const nodes = [
    'Video Input',
    'Decoder',
    'Detector (YOLOX)',
    'Tracker (ByteTrack)',
    'Homography',
    'Speed',
    'Uncertainty',
    'Evidence'
  ];

  return (
    <div>
      <div className="card">
        <h3>CV pipeline</h3>
        <div className="flow">
          {nodes.map((n, i) => (
            <React.Fragment key={i}>
              <span className="node">{n}</span>
              {i < nodes.length - 1 && <span className="mu">➔</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="card">
          <h3>API playground</h3>
          <select
            style={{ width: 'auto' }}
            value={selectedEp}
            onChange={e => setSelectedEp(e.target.value)}
          >
            {Object.keys(EP).map(k => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          <h3 style={{ marginTop: 10 }}>cURL</h3>
          <pre>{curlStr}</pre>
          <h3>Response</h3>
          <pre>{JSON.stringify(liveResponse, null, 2)}</pre>
        </div>

        <div className="card">
          <h3>Database schema</h3>
          <pre>
{`projects(id PK, name, description, created_at)
videos(id PK, project_id FK, filename, storage_path, sha256_hash, duration_sec, fps, width, height)
calibrations(id PK, video_id FK, version, h_matrix_json, image_points_json, world_points_json, reprojection_rmse_m)
processing_jobs(id PK, video_id FK, calibration_id FK, status, stage, progress_pct, error_message)
tracks(id PK, job_id FK, track_id, vehicle_class, confidence, first_frame, last_frame, trajectory_json)
speed_measurements(id PK, track_id FK, frame_index, timestamp, instantaneous_kmh, smoothed_kmh, uncertainty_kmh)
violations(id PK, job_id FK, track_id, vehicle_class, frame_index, timestamp, estimated_speed_kmh, review_status)
evidence(id PK, violation_id FK, full_frame_path, crop_frame_path, metadata_json)
experiments(id PK, name, dataset_name, detector_name, tracker_name, speed_method, mae_kmh, rmse_kmh, r2_score)`}
          </pre>
        </div>
      </div>
    </div>
  );
};
