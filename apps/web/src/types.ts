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
  time: string | number;
  trackId: number;
  vehicleClass: string;
  speedKmh: number;
  limitKmh: number;
  appliedToleranceKmh?: number;
  effectiveThresholdKmh?: number;
  amountOverThresholdKmh?: number;
  uncertaintyKmh: number;
  location: string;
  status: 'PENDING' | 'ACCEPTED' | 'REJECTED';
  reviewerNotes?: string;
  fullFrameUrl?: string;
  cropFrameUrl?: string;
  evidenceAvailable?: boolean;
  evidenceReason?: string;
  jobId?: number;
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

// ===========================================================================
// Universal Intelligence & Multi-Domain Interfaces
// ===========================================================================

export type AnalysisModule = 'traffic' | 'examination';

export type SceneType = 'TRAFFIC' | 'CLASSROOM_EXAMINATION' | 'OTHER_UNKNOWN' | 'AMBIGUOUS';

export interface SceneClassificationResult {
  scene_type: SceneType;
  confidence: number;
  recommended_module: AnalysisModule;
  requires_user_confirmation: boolean;
  reason: string;
  model_version: string;
  sampling_strategy?: string;
  scores?: Record<string, number>;
}

export type ExaminationEventType =
  | 'POSSIBLE_PHONE_INTERACTION'
  | 'RESTRICTED_ZONE_ENTRY'
  | 'LEAVING_SEATING_ZONE'
  | 'POSSIBLE_OBJECT_TRANSFER'
  | 'REPEATED_ORIENTATION_PATTERN'
  | 'OTHER_OBSERVABLE_INTERACTION';

export interface ExaminationEvent {
  id?: number;
  event_id: string;
  job_id: number;
  video_id: number;
  event_type: ExaminationEventType;
  start_frame: number;
  end_frame?: number;
  start_time: number;
  end_time?: number;
  person_track_id: number;
  associated_object_track_id?: number;
  associated_object_class?: string;
  supporting_objects?: Array<{ class_name: string; confidence: number; bbox?: number[] }> | string[];
  supporting_frames?: number[];
  confidence: number;
  uncertainty: number;
  priority?: 'RED' | 'AMBER' | 'INFO';
  evidence_strength?: 'HIGH' | 'MEDIUM' | 'LOW';
  detection_confidence?: number;
  association_confidence?: number;
  event_confidence?: number;
  uncertainty_reasons?: string[];
  primary_finding?: boolean;
  supporting_observations?: string;
  visibility_warnings?: string[];
  description: string;
  review_status: 'PENDING' | 'ACCEPTED' | 'REJECTED';
  reviewer_notes?: string;
}

export interface ExaminationZone {
  zone_id: string;
  zone_name: string;
  zone_type: 'SEATING_DESK' | 'SEATING_ROW' | 'RESTRICTED_PODIUM' | 'AISLE' | 'CORRIDOR';
  polygon: [number, number][];
}

export interface ExaminationPersonTrack {
  track_id: number;
  first_frame: number;
  last_frame: number;
  assigned_zone?: string;
  trajectory?: Array<{ frame: number; frame_index?: number; bbox: number[] }>;
}

export interface GeminiStatus {
  enabled: boolean;
  configured: boolean;
  status: 'READY' | 'KEY_MISSING' | 'DISABLED' | 'ERROR';
  model: string;
  max_selected_frames: number;
  timeout_sec: number;
  detail: string;
}

export interface GeminiCandidateInterpretation {
  event_id: string;
  event_type: string;
  observed_behaviour: string;
  contextual_assessment: string;
  supporting_frame_references: number[];
  uncertainty_notes: string[];
  visibility_warnings: string[];
  recommended_review_priority: string;
  heuristic_relevance_score: number;
}

export interface GeminiExamAnalysis {
  scene_summary: string;
  environment_type: string;
  candidate_findings: GeminiCandidateInterpretation[];
  contradictory_evidence_notes: string[];
  primary_finding_event_id?: string;
  proctor_action_recommendation: string;
  model_identifier: string;
}

