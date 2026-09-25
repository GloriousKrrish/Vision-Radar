export interface VehicleTrack {
  id: number;
  cls: string;
  confidence: number;
  speedKmh: number;
  smoothedKmh: number;
  uncertaintyKmh: number;
  errorComponents: {
    homography_perspective_pct: number;
    centroid_jitter_pct: number;
    timestamp_variance_pct: number;
  };
  plate: string;
}

export interface CandidateViolation {
  id: number;
  time: string;
  trackId: number;
  vehicleClass: string;
  speedKmh: number;
  limitKmh: number;
  uncertaintyKmh: number;
  location: string;
  status: 'PENDING' | 'ACCEPTED' | 'REJECTED';
  fullFrameUrl?: string;
  cropFrameUrl?: string;
}

export interface CalibrationData {
  hMatrix: number[][];
  imagePoints: [number, number][];
  worldPoints: [number, number][];
  reprojectionRmseM: number;
  scaleX: number;
  scaleY: number;
  cameraHeightM: number;
  pitchDeg: number;
}

export interface ExperimentMetric {
  id: number;
  name: string;
  datasetName: string;
  detectorName: string;
  trackerName: string;
  speedMethod: string;
  maeKmh: number;
  rmseKmh: number;
  r2Score: number;
}
