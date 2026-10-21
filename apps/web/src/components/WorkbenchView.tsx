import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiUrl, getWsUrl } from '../config/api';
import { AnalysisModule, SceneClassificationResult, ExaminationEvent, ExaminationPersonTrack, ExaminationZone, GeminiStatus, GeminiExamAnalysis, ProcessingMode } from '../types';
import {
  SceneRecommendationBanner,
  PrimaryFindingPanel,
  GeminiContextualPanel,
  ExaminationOverviewCard,
  ExaminationEventsTable,
  ExaminationSidebarInspector
} from './ExaminationComponents';
import { useSharedJob } from '../context/SharedJobContext';

// ===========================================================================
// Constants & Canvas Resolution
// ===========================================================================
const W = 800;
const H = 450;
const FPS = 30;
const N = 900;

const RT_BUFFER_FRAMES = 120; // Sliding window of streaming frames

// Synthetic demo vehicle data — used ONLY in Demo Mode, NEVER during real video
const CL = ['Sedan', 'SUV', 'Truck', 'Motorcycle'];
const S_fn = (d: number) => 1 / (1 + d / 25);
const proj = (lx: number, d: number): [number, number] => [400 + (lx - 6) * S_fn(d) * 60, 110 + 330 * S_fn(d)];
function rnd(s: number) { return () => (s = (s * 16807) % 2147483647) / 2147483647; }
const R = rnd(7);
const V: any[] = [];
for (let i = 0; i < 14; i++) {
  const c = CL[i % 4 === 3 && i % 8 !== 3 ? 0 : i % 4];
  V.push({
    id: i + 1, cls: c, lx: [2, 6, 10][i % 3],
    v: (58 + R() * 55) / 3.6, off: R() * 150, conf: 0.82 + R() * 0.16,
    w: c === 'Truck' ? 2.5 : c === 'Motorcycle' ? 0.9 : 1.9,
    h: c === 'Truck' ? 3.2 : c === 'Motorcycle' ? 1.5 : 1.5,
    plate: 'MH' + (12 + i) + ' ' + String.fromCharCode(65 + i, 66 + i) + ' ' + (1000 + (i * 371) % 9000)
  });
}
const dOf = (a: any, t: number) => 140 - ((a.off + a.v * t) % 150);
const CP = [[0, 10], [12, 10], [12, 90], [0, 90]];
let HP = [[0, 10], [12, 10], [12, 90], [0, 90]].map(([a, b]) => proj(a, b));
function solve(A: number[][], b: number[]) {
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let m = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[m][i])) m = r;
    [A[i], A[m]] = [A[m], A[i]]; [b[i], b[m]] = [b[m], b[i]];
    for (let r = i + 1; r < n; r++) { const k = A[r][i] / A[i][i]; for (let c = i; c < n; c++) A[r][c] -= k * A[i][c]; b[r] -= k * b[i]; }
  }
  const x = Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c]; x[i] = s / A[i][i]; }
  return x;
}
let Hm: number[][] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
function calcH() {
  const A: number[][] = []; const b: number[] = [];
  HP.forEach(([x, y], i) => { const [X, Y] = CP[i]; A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.push(X); A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.push(Y); });
  const h = solve(A, b); Hm = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}
const toW = (x: number, y: number) => { const q = Hm[2][0] * x + Hm[2][1] * y + 1; return [(Hm[0][0] * x + Hm[0][1] * y + Hm[0][2]) / q, (Hm[1][0] * x + Hm[1][1] * y + Hm[1][2]) / q]; };
function state(a: any, t: number) { const d = dOf(a, t); const [x, y] = proj(a.lx, d); const s = S_fn(d); return { d, x, y, s, bw: a.w * 60 * s, bh: a.h * 60 * s }; }
function jit(a: any, fr: number, k: number) { return Math.sin(a.id * 12.9 + fr * 78.2 + k * 3.1) * 0.7; }
function estSpeed(a: any, fr: number) {
  const dt = 15; const p = (q: number) => { const st = state(a, q / FPS); return toW(st.x + jit(a, q, 0), st.y + jit(a, q, 1)); };
  const A = p(fr); const B = p(fr - dt); return (Math.hypot(A[0] - B[0], A[1] - B[1]) / (dt / FPS)) * 3.6;
}

// ===========================================================================
// Authoritative Pipeline State Machine (Requirement #8)
// ===========================================================================
export type PipelineStage =
  | 'IDLE'
  | 'VIDEO_READY'        // Video visible locally
  | 'UPLOADING'          // File upload in progress
  | 'UPLOAD_COMPLETE'    // Video file saved on backend
  | 'JOB_CREATED'        // Job ID returned, connecting WebSocket
  | 'PROCESSING'         // Worker initialized
  | 'LIVE_INFERENCE'     // Active WebSocket streaming frame results
  | 'FINALIZING'         // Worker persisting trajectories to DB
  | 'COMPLETED'          // Final DB tracks loaded
  | 'ERROR';

// ===========================================================================
// WorkbenchView Component
// ===========================================================================
export const WorkbenchView: React.FC = () => {
  const { activeJobId: sharedJobId, selectJob: selectSharedJob, openExportModal } = useSharedJob();

  // --- playback state ---
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);

  // --- demo/real mode ---
  const [isRealVideo, setIsRealVideo] = useState(true);
  const [srcName, setSrcName] = useState('Traffic1.mp4');
  const [videoSrc, setVideoSrc] = useState<string | null>(null);

  // --- job lifecycle & telemetry ---
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('IDLE');
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const [jobStatusText, setJobStatusText] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);

  // --- real-time telemetry ---
  const [liveTrackCount, setLiveTrackCount] = useState(0);
  const [latestInferenceFrame, setLatestInferenceFrame] = useState<number | null>(null);
  const [firstDetectionMs, setFirstDetectionMs] = useState<number | null>(null);
  const [firstBoxMs, setFirstBoxMs] = useState<number | null>(null);
  const [processingFps, setProcessingFps] = useState<number | null>(null);
  const [realTimeFactor, setRealTimeFactor] = useState<number | null>(null);

  // --- track & calibration data ---
  const [realTracks, setRealTracks] = useState<any[]>([]);
  const [telemetry, setTelemetry] = useState<any | null>(null);
  const [realCalib, setRealCalib] = useState<any | null>(null);
  const [activeVideoId, setActiveVideoId] = useState<number | null>(null);
  const [tableFilter, setTableFilter] = useState<'ALL' | 'SUSTAINED' | 'IN_FRAME'>('ALL');

  // --- inspector & live log ---
  const [selTrack, setSelTrack] = useState<number | null>(null);
  const [violations, setViolations] = useState<any[]>([]);
  const [speedLimit, setSpeedLimit] = useState(80);
  const [speedLimitInput, setSpeedLimitInput] = useState<number>(80);
  const [warningToleranceInput, setWarningToleranceInput] = useState<number>(0);
  const [appliedSpeedLimit, setAppliedSpeedLimit] = useState<number>(80);
  const [appliedTolerance, setAppliedTolerance] = useState<number>(0);
  const [isConfigUnsaved, setIsConfigUnsaved] = useState<boolean>(false);
  const [speedConfigError, setSpeedConfigError] = useState<string | null>(null);
  const [activeEvidenceModal, setActiveEvidenceModal] = useState<any | null>(null);
  const [sidebarTab, setSidebarTab] = useState<'log' | 'inspector'>('log');
  const [logFilter, setLogFilter] = useState<'all' | 'active' | 'speeding'>('all');
  const [renderTick, setRenderTick] = useState<number>(0);

  // --- Universal Multi-Domain & Scene Routing State ---
  const [activeModule, setActiveModule] = useState<AnalysisModule>('traffic');
  const [sceneRec, setSceneRec] = useState<(SceneClassificationResult & { video_id: number; video_name: string }) | null>(null);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [examTracks, setExamTracks] = useState<ExaminationPersonTrack[]>([]);
  const [examEvents, setExamEvents] = useState<ExaminationEvent[]>([]);
  const [examZones, setExamZones] = useState<ExaminationZone[]>([]);
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [primaryExamFinding, setPrimaryExamFinding] = useState<ExaminationEvent | null>(null);
  const [geminiStatus, setGeminiStatus] = useState<GeminiStatus | null>(null);
  const [geminiExamAnalysis, setGeminiExamAnalysis] = useState<GeminiExamAnalysis | null>(null);
  const [isAnalyzingGemini, setIsAnalyzingGemini] = useState<boolean>(false);
  const [geminiError, setGeminiError] = useState<string | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('balanced');

  // --- crowded-scene intelligent visualization state ---
  const [vizMode, setVizMode] = useState<'MINIMAL' | 'STANDARD' | 'INTELLIGENCE'>('STANDARD');
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showTrajectories, setShowTrajectories] = useState<boolean>(false);
  const [classFilter, setClassFilter] = useState<'ALL' | 'CAR' | 'BUS' | 'TRUCK' | 'MOTORCYCLE'>('ALL');
  const [minConfidence, setMinConfidence] = useState<number>(0.25);
  const [focusSelected, setFocusSelected] = useState<boolean>(false);
  const [maxLabels, setMaxLabels] = useState<number>(14);
  const [hoveredTrack, setHoveredTrack] = useState<number | null>(null);
  const hoveredTrackRef = useRef<number | null>(null);

  // --- table pagination ---
  const [tablePage, setTablePage] = useState<number>(1);
  const TABLE_PAGE_SIZE = 10;

  // --- refs ---
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const seenRef = useRef<Set<number>>(new Set());
  const wsRef = useRef<WebSocket | null>(null);

  // Bounded Real-Time Frame Buffer: Map<frame_index, track[]>
  const rtFrameBufferRef = useRef<Map<number, any[]>>(new Map());

  // Finalized Frame Map (from DB tracks after completion)
  const frameMapRef = useRef<Map<number, any[]>>(new Map());

  // Timestamps for latency
  const uploadStartRef = useRef<number>(0);
  const firstBoxRenderedRef = useRef<boolean>(false);
  const isLiveStreamingRef = useRef<boolean>(false);

  const isRealVideoRef = useRef<boolean>(true);
  const pipelineStageRef = useRef<PipelineStage>('IDLE');
  const activeJobIdRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number>(-1);

  useEffect(() => { isRealVideoRef.current = isRealVideo; }, [isRealVideo]);
  useEffect(() => { pipelineStageRef.current = pipelineStage; }, [pipelineStage]);
  useEffect(() => { activeJobIdRef.current = activeJobId; }, [activeJobId]);

  useEffect(() => { calcH(); }, []);

  // =========================================================================
  // Auto-load last succeeded job on startup (if idle)
  // =========================================================================
  useEffect(() => {
    if (pipelineStage !== 'IDLE') return;
    const selectCompletedJob = async () => {
      try {
        const res = await fetch(apiUrl('/api/v1/jobs?status=SUCCEEDED'));
        if (!res.ok) return;
        const jobs = await res.json();
        if (!jobs?.length) return;
        let job = null;
        for (const j of jobs) {
          try {
            // Verify video stream is physically available on disk (HEAD request)
            const check = await fetch(apiUrl(`/api/v1/videos/${j.video_id}/stream`), { method: 'HEAD' });
            if (check.ok) {
              job = j;
              break;
            }
          } catch {}
        }
        if (!job) return;

        setIsRealVideo(true);
        isRealVideoRef.current = true;
        setActiveJobId(job.id);
        activeJobIdRef.current = job.id;
        setActiveVideoId(job.video_id);
        setJobStatusText(`Job #${job.id} Complete — REAL VIDEO · LIVE AI`);
        setPipelineStage('COMPLETED');
        if (job.telemetry) setTelemetry(job.telemetry);

        rtFrameBufferRef.current.clear();
        frameMapRef.current.clear();

        const streamUrl = apiUrl(`/api/v1/videos/${job.video_id}/stream`);
        setVideoSrc(streamUrl);
        if (videoRef.current) {
          videoRef.current.muted = true;
          videoRef.current.defaultMuted = true;
          videoRef.current.src = streamUrl;
          videoRef.current.loop = true;
          videoRef.current.load();
          videoRef.current.play().catch(() => {});
        }

        let startupModule: AnalysisModule = 'traffic';
        if (job.config_json) {
          try {
            const parsed = typeof job.config_json === 'string' ? JSON.parse(job.config_json) : job.config_json;
            if (parsed?.module === 'examination') startupModule = 'examination';
          } catch {}
        }
        setActiveModule(startupModule);

        const [trkRes, calibRes, vidInfoRes, examRes, zonesRes] = await Promise.all([
          startupModule === 'traffic' ? fetch(apiUrl(`/api/v1/jobs/${job.id}/tracks`)).catch(() => null) : Promise.resolve(null),
          startupModule === 'traffic' ? fetch(apiUrl(`/api/v1/videos/${job.video_id}/calibrations`)).catch(() => null) : Promise.resolve(null),
          fetch(apiUrl(`/api/v1/videos/${job.video_id}`)).catch(() => null),
          startupModule === 'examination' ? fetch(apiUrl(`/api/v1/jobs/${job.id}/examination/results`)).catch(() => null) : Promise.resolve(null),
          startupModule === 'examination' ? fetch(apiUrl(`/api/v1/videos/${job.video_id}/examination/zones`)).catch(() => null) : Promise.resolve(null)
        ]);

        if (vidInfoRes && vidInfoRes.ok) {
          const vData = await vidInfoRes.json();
          if (vData.filename) setSrcName(vData.filename);
          setTelemetry((prev: any) => ({
            ...(prev || {}),
            source_width: vData.width || prev?.source_width || 1920,
            source_height: vData.height || prev?.source_height || 1080,
            video_fps: vData.fps || prev?.video_fps || 29.97,
            video_duration_sec: vData.duration_sec || prev?.video_duration_sec || 0.0,
            total_frames: prev?.total_frames || Math.floor((vData.duration_sec || 0) * (vData.fps || 29.97))
          }));
        }

        if (startupModule === 'traffic') {
          if (trkRes && trkRes.ok) {
            const trkData = await trkRes.json();
            if (trkData?.length) {
              setRealTracks(trkData);
            }
          }
          if (calibRes && calibRes.ok) {
            const calibData = await calibRes.json();
            if (calibData?.length) setRealCalib(calibData[0]);
          }
        } else {
          if (examRes && examRes.ok) {
            const exData = await examRes.json();
            setExamTracks(exData.tracks || []);
            setExamEvents(exData.events || []);
            if (exData.primary_finding) setPrimaryExamFinding(exData.primary_finding);
            if (exData.zones) setExamZones(exData.zones);
            const fMap = new Map<number, any[]>();
            (exData.tracks || []).forEach((t: any) => {
              (t.trajectory || []).forEach((h: any) => {
                const f = h.frame;
                const bbox = h.bbox;
                const det = {
                  track_id: t.track_id,
                  entity_type: 'person',
                  rx: bbox[0],
                  ry: bbox[1],
                  rw: bbox[2] - bbox[0],
                  rh: bbox[3] - bbox[1],
                  confidence: 0.9,
                  assigned_zone: t.assigned_zone
                };
                if (!fMap.has(f)) fMap.set(f, []);
                fMap.get(f)!.push(det);
              });
            });
            frameMapRef.current = fMap;
          }
          if (zonesRes && zonesRes.ok) {
            const zData = await zonesRes.json();
            setExamZones(zData || []);
          }
        }
      } catch (e) {
        console.error('Auto-load failed:', e);
      }
    };
    selectCompletedJob();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // =========================================================================
  // Build finalized frame map from DB tracks
  // =========================================================================
  useEffect(() => {
    const map = new Map<number, any[]>();
    realTracks.forEach((trk: any) => {
      const traj = trk.trajectory || [];
      // Pre-compute avg/max speed from speed_measurements for this track
      const validSpeeds = (trk.speed_measurements || [])
        .map((s: any) => s.smoothed_kmh ?? s.instantaneous_kmh)
        .filter((v: any) => v != null && v > 0);
      const trkAvgSpeed = validSpeeds.length > 0
        ? Math.round(validSpeeds.reduce((a: number, b: number) => a + b, 0) / validSpeeds.length * 10) / 10
        : null;
      const trkMaxSpeed = validSpeeds.length > 0 ? Math.round(Math.max(...validSpeeds) * 10) / 10 : null;
      const trkMinSpeed = validSpeeds.length > 0 ? Math.round(Math.min(...validSpeeds) * 10) / 10 : null;
      traj.forEach((pt: any) => {
        const f = pt.frame_index;
        if (f === undefined || f === null) return;
        const rawBbox = pt.bbox;
        if (!rawBbox || rawBbox.length < 4) return;
        let [rx, ry, rw, rh] = rawBbox;
        if (rw > rx) rw = rw - rx;
        if (rh > ry) rh = rh - ry;
        const speedInfo = trk.speed_measurements?.find(
          (s: any) => s.frame === f || s.frame_index === f
        );
        let speedKmh = speedInfo?.smoothed_kmh ?? speedInfo?.instantaneous_kmh ?? null;
        let speedStatus = speedInfo?.error_components?.validity;
        let speedKmhCalibrated = (speedKmh !== null && speedKmh > 0 && (speedStatus === 'VALID' || !speedStatus)) ? speedKmh : null;
        let measurementState: 'CALIBRATED_SPEED' | 'UNCALIBRATED_MOTION' | 'SPEED_UNAVAILABLE' = 'SPEED_UNAVAILABLE';
        let motionPxS: number | null = null;

        // Optical displacement motion fallback (Category C: uncalibrated assumed scale)
        if ((speedKmh === null || speedKmh <= 0) && traj.length >= 2) {
          const ptIdx = traj.findIndex((p: any) => p.frame_index === f);
          if (ptIdx >= 1) {
            const prevPt = traj[Math.max(0, ptIdx - 4)];
            const b1 = pt.bbox;
            const b0 = prevPt.bbox;
            if (b1 && b0 && b1.length >= 4 && b0.length >= 4) {
              const cx1 = (b1[0] + (b1[2] > b1[0] ? b1[2] : b1[0] + b1[2])) / 2;
              const cy1 = (b1[1] + (b1[3] > b1[1] ? b1[3] : b1[1] + b1[3])) / 2;
              const cx0 = (b0[0] + (b0[2] > b0[0] ? b0[2] : b0[0] + b0[2])) / 2;
              const cy0 = (b0[1] + (b0[3] > b0[1] ? b0[3] : b0[1] + b0[3])) / 2;
              const dist = Math.hypot(cx1 - cx0, cy1 - cy0);
              const df = Math.max(1, f - prevPt.frame_index);
              const dt = df / (telemetry?.video_fps || 29.97);
              if (dt > 0) {
                motionPxS = Math.round(dist / dt);
                const est = Math.round((dist * 0.05 / dt) * 3.6);
                if (est > 5 && est < 220) {
                  speedKmh = est;
                  speedStatus = 'UNCALIBRATED_MOTION';
                  measurementState = 'UNCALIBRATED_MOTION';
                }
              }
            }
          }
        }

        if (speedKmhCalibrated !== null && speedKmhCalibrated > 0) {
          measurementState = 'CALIBRATED_SPEED';
          speedStatus = 'VALID';
        } else if (measurementState !== 'UNCALIBRATED_MOTION') {
          measurementState = 'SPEED_UNAVAILABLE';
          if (!speedStatus) {
            speedStatus = realCalib ? 'CALCULATING...' : 'CALIBRATION_REQUIRED';
          }
        }

        if (!map.has(f)) map.set(f, []);
        map.get(f)!.push({
          track_id: trk.track_id, vehicle_class: trk.vehicle_class,
          confidence: trk.confidence, rx, ry, rw, rh,
          speedKmh,
          speedKmhCalibrated,
          measurementState,
          motionPxS,
          speedStatus,
          avgSpeedKmh: trkAvgSpeed, maxSpeedKmh: trkMaxSpeed, minSpeedKmh: trkMinSpeed,
          firstFrame: trk.first_frame, lastFrame: trk.last_frame,
          detectionCount: traj.length
        });
      });
    });
    frameMapRef.current = map;
    if (map.size > 0) {
      rtFrameBufferRef.current.clear();
    }
  }, [realTracks]);

  const lastRenderedFrameIdRef = useRef<number>(-1);
  const [serverPipelineLatencyMs, setServerPipelineLatencyMs] = useState<number | null>(null);
  const [perceptionMetrics, setPerceptionMetrics] = useState<any | null>(null);

  const loadJobResults = useCallback(async (jobId: number) => {
    if (activeJobIdRef.current !== null && activeJobIdRef.current !== jobId) {
      return; // Stale job rejection
    }
    try {
      const [trkRes, violsRes, jobRes] = await Promise.all([
        fetch(apiUrl(`/api/v1/jobs/${jobId}/tracks`)).catch(() => null),
        fetch(apiUrl(`/api/v1/violations?job_id=${jobId}&limit=30`)).catch(() => null),
        fetch(apiUrl(`/api/v1/jobs/${jobId}`)).catch(() => null)
      ]);

      if (activeJobIdRef.current !== null && activeJobIdRef.current !== jobId) {
        return; // Stale job rejection after async response
      }

      const trkData = (trkRes && trkRes.ok) ? await trkRes.json() : [];
      const violsData = (violsRes && violsRes.ok) ? await violsRes.json() : [];
      const jobData = (jobRes && jobRes.ok) ? await jobRes.json() : null;

      let jobModule: AnalysisModule = 'traffic';
      if (jobData?.config_json) {
        try {
          const cfg = typeof jobData.config_json === 'string' ? JSON.parse(jobData.config_json) : jobData.config_json;
          if (cfg?.module === 'examination') jobModule = 'examination';
          if (cfg?.speed_limit_kmh !== undefined && cfg?.speed_limit_kmh !== null) {
            const lim = Number(cfg.speed_limit_kmh) || 80;
            const tol = Number(cfg.warning_tolerance_kmh || 0);
            setSpeedLimit(lim);
            setSpeedLimitInput(lim);
            setAppliedSpeedLimit(lim);
            setWarningToleranceInput(tol);
            setAppliedTolerance(tol);
            setIsConfigUnsaved(false);
          }
        } catch {}
      }
      setActiveModule(jobModule);

      if (jobModule === 'traffic') {
        if (trkData?.length) {
          setRealTracks(trkData);
        }
        if (violsData?.length) {
          setViolations(violsData.map((v: any) => ({
            id: v.id,
            jobId: v.job_id,
            trackId: v.track_id,
            vehicleClass: v.vehicle_class,
            time: v.timestamp,
            speedKmh: v.estimated_speed_kmh,
            limitKmh: v.speed_limit_kmh,
            appliedToleranceKmh: v.applied_tolerance_kmh ?? 0,
            effectiveThresholdKmh: v.effective_threshold_kmh ?? (v.speed_limit_kmh + (v.applied_tolerance_kmh ?? 0)),
            amountOverThresholdKmh: v.amount_over_threshold_kmh ?? Math.max(0, v.estimated_speed_kmh - v.speed_limit_kmh),
            uncertaintyKmh: v.uncertainty_kmh,
            location: v.location_label,
            status: v.review_status,
            reviewerNotes: v.reviewer_notes,
            fullFrameUrl: v.full_frame_url,
            cropFrameUrl: v.crop_frame_url,
            evidenceAvailable: v.evidence_available !== false && Boolean(v.full_frame_url || v.crop_frame_url),
            evidenceReason: v.evidence_reason,
            t: v.timestamp,
            v: v.estimated_speed_kmh,
            loc: v.location_label
          })));
        } else {
          setViolations([]);
        }
      } else {
        try {
          const [examRes, zonesRes] = await Promise.all([
            fetch(apiUrl(`/api/v1/jobs/${jobId}/examination/results`)).catch(() => null),
            jobData ? fetch(apiUrl(`/api/v1/videos/${jobData.video_id}/examination/zones`)).catch(() => null) : Promise.resolve(null)
          ]);
          if (examRes && examRes.ok) {
            const exData = await examRes.json();
            setExamTracks(exData.tracks || []);
            setExamEvents(exData.events || []);
            if (exData.primary_finding) setPrimaryExamFinding(exData.primary_finding);
            if (exData.zones) setExamZones(exData.zones);
            const fMap = new Map<number, any[]>();
            (exData.tracks || []).forEach((t: any) => {
              (t.trajectory || []).forEach((h: any) => {
                const f = h.frame;
                const bbox = h.bbox;
                const det = {
                  track_id: t.track_id,
                  entity_type: t.entity_type || 'person',
                  rx: bbox[0],
                  ry: bbox[1],
                  rw: bbox[2] - bbox[0],
                  rh: bbox[3] - bbox[1],
                  confidence: h.conf ?? 0.9,
                  assigned_zone: t.assigned_zone,
                  assigned_role: t.assigned_role,
                  role_source: t.role_source
                };
                if (!fMap.has(f)) fMap.set(f, []);
                fMap.get(f)!.push(det);
              });
            });
            frameMapRef.current = fMap;
          }
          if (zonesRes && zonesRes.ok) {
            const zData = await zonesRes.json();
            setExamZones(zData || []);
          }
        } catch (e) {
          console.warn('Examination load error:', e);
        }
      }

      if (jobData?.telemetry) {
        setTelemetry(jobData.telemetry);
        if (jobData.telemetry.average_fps) setProcessingFps(jobData.telemetry.average_fps);
        if (jobData.telemetry.real_time_factor) setRealTimeFactor(jobData.telemetry.real_time_factor);
      }
      setPipelineStage('COMPLETED');
      setJobStatusText(`Job #${jobId} COMPLETED — REAL VIDEO · LIVE AI ANALYSIS [${jobModule.toUpperCase()}]`);
    } catch (err) {
      console.error('Failed to load job results:', err);
      setPipelineStage('COMPLETED');
      setJobStatusText(`Job #${jobId} Complete — REAL VIDEO · LIVE AI ANALYSIS`);
    }
  }, []);

  // Rest Polling Fallback to guarantee job state synchronization if WebSocket disconnects or finishes fast
  useEffect(() => {
    if (sharedJobId && sharedJobId !== activeJobId && pipelineStage !== 'LIVE_INFERENCE' && pipelineStage !== 'UPLOADING') {
      setActiveJobId(sharedJobId);
      activeJobIdRef.current = sharedJobId;
      loadJobResults(sharedJobId);
    }
  }, [sharedJobId, activeJobId, pipelineStage, loadJobResults]);

  useEffect(() => {
    if (!activeJobId || pipelineStage === 'COMPLETED' || pipelineStage === 'ERROR' || pipelineStage === 'IDLE') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(apiUrl(`/api/v1/jobs/${activeJobId}`));
        if (!res.ok) return;
        const job = await res.json();
        if (job.status === 'SUCCEEDED') {
          isLiveStreamingRef.current = false;
          loadJobResults(activeJobId);
        } else if (job.status === 'FAILED') {
          isLiveStreamingRef.current = false;
          setPipelineStage('ERROR');
          setJobStatusText(`Error on Job #${activeJobId}: ${job.error_message || 'Job processing failed'}`);
        } else if (job.progress_pct > 0) {
          setJobProgress(job.progress_pct);
          if (pipelineStage === 'JOB_CREATED' || pipelineStage === 'UPLOADING') {
            setPipelineStage('PROCESSING');
            setJobStatusText(`Job #${activeJobId} in progress (${Math.round(job.progress_pct)}%)...`);
          }
        }
      } catch (e) {
        console.error('Job polling error:', e);
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [activeJobId, pipelineStage, loadJobResults]);

  // --- Gemini Multimodal Integration Effects & Handlers ---
  useEffect(() => {
    fetch(apiUrl('/api/v1/gemini/status'))
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) setGeminiStatus(data);
      })
      .catch(err => console.warn('Failed to load Gemini status:', err));
  }, []);

  useEffect(() => {
    if (activeJobId && activeModule === 'examination') {
      fetch(apiUrl(`/api/v1/jobs/${activeJobId}/examination/gemini-analysis`))
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data && data.status === 'SUCCESS' && data.analysis) {
            setGeminiExamAnalysis(data.analysis);
          } else {
            setGeminiExamAnalysis(null);
          }
        })
        .catch(() => setGeminiExamAnalysis(null));
    }
  }, [activeJobId, activeModule]);

  const handleTriggerGeminiAnalysis = async (forceRefresh: boolean = false) => {
    if (!activeJobId) return;
    try {
      setIsAnalyzingGemini(true);
      setGeminiError(null);
      const res = await fetch(apiUrl(`/api/v1/jobs/${activeJobId}/examination/gemini-analyze`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force_refresh: forceRefresh })
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      if (data.status === 'SUCCESS' && data.analysis) {
        setGeminiExamAnalysis(data.analysis);
      } else {
        throw new Error(data.detail || 'Contextual analysis returned unexpected status');
      }
    } catch (e: any) {
      setGeminiError(e.message || String(e));
    } finally {
      setIsAnalyzingGemini(false);
    }
  };

  // =========================================================================
  // Speed Limit & Violations Configuration Handler (Phase 3)
  // =========================================================================
  const handleApplySpeedLimit = async () => {
    setSpeedConfigError(null);
    if (!speedLimitInput || speedLimitInput <= 0) {
      setSpeedConfigError('Speed limit must be greater than 0 km/h.');
      return;
    }
    if (warningToleranceInput < 0) {
      setSpeedConfigError('Warning tolerance must be non-negative (>= 0 km/h).');
      return;
    }

    const currentJobId = activeJobId || telemetry?.job_id;
    if (!currentJobId) {
      setAppliedSpeedLimit(speedLimitInput);
      setAppliedTolerance(warningToleranceInput);
      setSpeedLimit(speedLimitInput);
      setIsConfigUnsaved(false);
      seenRef.current.clear();
      return;
    }

    try {
      const res = await fetch(apiUrl(`/api/v1/jobs/${currentJobId}/speed-limit`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speed_limit_kmh: speedLimitInput,
          warning_tolerance_kmh: warningToleranceInput
        })
      });
      if (res.ok) {
        const data = await res.json();
        setAppliedSpeedLimit(data.speed_limit_kmh);
        setAppliedTolerance(data.warning_tolerance_kmh);
        setSpeedLimit(data.speed_limit_kmh);
        setIsConfigUnsaved(false);
        seenRef.current.clear();

        const vRes = await fetch(apiUrl(`/api/v1/violations?job_id=${currentJobId}&limit=50`));
        if (vRes.ok) {
          const vData = await vRes.json();
          setViolations(vData.map((v: any) => ({
            id: v.id,
            jobId: v.job_id,
            trackId: v.track_id,
            vehicleClass: v.vehicle_class,
            time: v.timestamp,
            speedKmh: v.estimated_speed_kmh,
            limitKmh: v.speed_limit_kmh,
            appliedToleranceKmh: v.applied_tolerance_kmh ?? 0,
            effectiveThresholdKmh: v.effective_threshold_kmh ?? (v.speed_limit_kmh + (v.applied_tolerance_kmh ?? 0)),
            amountOverThresholdKmh: v.amount_over_threshold_kmh ?? Math.max(0, v.estimated_speed_kmh - v.speed_limit_kmh),
            uncertaintyKmh: v.uncertainty_kmh,
            location: v.location_label,
            status: v.review_status,
            reviewerNotes: v.reviewer_notes,
            fullFrameUrl: v.full_frame_url,
            cropFrameUrl: v.crop_frame_url,
            evidenceAvailable: v.evidence_available !== false && Boolean(v.full_frame_url || v.crop_frame_url),
            evidenceReason: v.evidence_reason,
            t: v.timestamp,
            v: v.estimated_speed_kmh,
            loc: v.location_label
          })));
        }
      } else {
        const err = await res.json();
        setSpeedConfigError(err.detail || 'Could not apply speed limit');
      }
    } catch (e: any) {
      setSpeedConfigError(`Network error applying speed limit: ${e.message}`);
    }
  };

  // Authoritative WebSocket Connection
  const connectWebSocket = useCallback((jobId: number) => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    lastRenderedFrameIdRef.current = -1;
    const wsUrl = getWsUrl(`/api/v1/jobs/${jobId}/stream`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    isLiveStreamingRef.current = true;

    ws.onopen = () => {
      setWsConnected(true);
      setJobStatusText(`Job #${jobId} — ● LIVE AI INFERENCE ACTIVE`);
      setPipelineStage('LIVE_INFERENCE');
    };

    ws.onmessage = (event) => {
      let msg: any;
      try { msg = JSON.parse(event.data); } catch { return; }

      if (activeJobIdRef.current !== null && activeJobIdRef.current !== jobId) {
        return; // Stale job isolation
      }

      const payload = msg.payload || msg;
      const type = msg.type;

      // Multi-Domain Envelope handling for Examination Intelligence
      const envelopeModule = msg.analysis_module || payload?.analysis_module;
      if (envelopeModule === 'examination' || activeModule === 'examination') {
        if (msg.result_type === 'frame_observation' || payload?.persons !== undefined) {
          const frameIdx: number = msg.frame_index ?? payload?.frame_index ?? -1;
          if (frameIdx >= 0 && frameIdx <= lastRenderedFrameIdRef.current) return;
          if (frameIdx >= 0) lastRenderedFrameIdRef.current = frameIdx;
          setLatestInferenceFrame(frameIdx);
          if (payload?.throughput_fps !== undefined) {
            setProcessingFps(payload.throughput_fps);
          }

          const persons = payload?.persons || [];
          const objects = payload?.objects || [];
          const activeEvts = payload?.active_events || [];
          const zones = payload?.zones || [];

          if (zones.length > 0) setExamZones(zones);

          if (activeEvts.length > 0) {
            setExamEvents(prev => {
              const existingIds = new Set(prev.map(e => e.event_id));
              const newEvts = activeEvts.filter((e: any) => !existingIds.has(e.event_id));
              return [...newEvts, ...prev].slice(0, 50);
            });
          }

          const examDets = [
            ...persons.map((p: any) => ({
              track_id: p.track_id,
              entity_type: 'person',
              rx: p.bbox[0],
              ry: p.bbox[1],
              rw: p.bbox[2] - p.bbox[0],
              rh: p.bbox[3] - p.bbox[1],
              confidence: p.confidence,
              assigned_zone: p.assigned_zone,
              assigned_role: p.assigned_role,
              role_source: p.role_source,
              zone_confidence: p.zone_confidence,
              is_occluded: p.is_occluded,
              active_events: activeEvts.filter((e: any) => e.person_track_id === p.track_id)
            })),
            ...objects.map((o: any, idx: number) => ({
              track_id: o.object_track_id || (10000 + idx),
              entity_type: o.object_class || o.class_name || 'object',
              rx: o.bbox[0],
              ry: o.bbox[1],
              rw: o.bbox[2] - o.bbox[0],
              rh: o.bbox[3] - o.bbox[1],
              confidence: o.confidence
            }))
          ];

          rtFrameBufferRef.current.set(frameIdx, examDets);
          setLiveTrackCount(persons.length);

          if (videoRef.current && videoRef.current.readyState > 1) {
            videoRef.current.playbackRate = 1.0;
          }
          return;
        } else if (msg.result_type === 'job_completed' || type === 'completed') {
          isLiveStreamingRef.current = false;
          setWsConnected(false);
          setPipelineStage('FINALIZING');
          setJobStatusText(`Job #${jobId} complete — loading examination results...`);
          loadJobResults(jobId);
          return;
        }
      }

      if (type === 'frame_result') {
        const frameIdx: number = payload.frame_id ?? payload.frame_index ?? -1;

        if (frameIdx >= 0 && frameIdx <= lastRenderedFrameIdRef.current) {
          return;
        }
        if (frameIdx >= 0) {
          lastRenderedFrameIdRef.current = frameIdx;
        }

        setLatestInferenceFrame(frameIdx);

        if (payload.capture_ts && payload.result_ts) {
          const latencyMs = Math.round((payload.result_ts - payload.capture_ts) * 1000);
          setServerPipelineLatencyMs(latencyMs);
        } else if (payload.server_pipeline_latency_ms !== undefined) {
          setServerPipelineLatencyMs(Math.round(payload.server_pipeline_latency_ms));
        }

        if (payload?.throughput_fps !== undefined) {
          setProcessingFps(payload.throughput_fps);
        }

        const tracks: any[] = payload.tracks || [];

        const renderTracks = tracks.map((t: any) => {
          const bbox = t.bbox ? (t.bbox.x1 !== undefined ? [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2] : t.bbox) : [0, 0, 0, 0];
          let [x1, y1, x2, y2] = bbox;
          const rx = Math.min(x1, x2);
          const ry = Math.min(y1, y2);
          const rw = Math.max(1, Math.abs(x2 - x1));
          const rh = Math.max(1, Math.abs(y2 - y1));
          const calibSpeed = t.speed_kmh_calibrated ?? (t.speed_status === 'VALID' ? t.speed_kmh : null);
          const measState: 'CALIBRATED_SPEED' | 'UNCALIBRATED_MOTION' | 'SPEED_UNAVAILABLE' =
            t.measurement_state ||
            (calibSpeed !== null && calibSpeed > 0 ? 'CALIBRATED_SPEED' :
             (t.speed_status === 'UNCALIBRATED_MOTION' || t.speed_status === 'UNCALIBRATED_ESTIMATE') ? 'UNCALIBRATED_MOTION' :
             'SPEED_UNAVAILABLE');

          return {
            track_id: t.track_id,
            vehicle_class: t.vehicle_class || 'Vehicle',
            confidence: t.confidence || 0,
            rx, ry, rw, rh,
            speedKmh: t.speed_kmh,
            speedKmhCalibrated: calibSpeed,
            speedStatus: t.speed_status || 'OUT_OF_ROI',
            measurementState: measState,
            motionPxS: t.motion_px_s,
            speedUncertainty: t.speed_uncertainty_kmh,
            avgSpeedKmh: t.avg_speed_kmh || null,
            maxSpeedKmh: t.max_speed_kmh || null,
            minSpeedKmh: t.min_speed_kmh || null,
            lane: t.lane || 'UNKNOWN',
            firstFrame: t.first_frame,
            lastFrame: t.last_frame,
            detectionCount: t.detection_count || 0,
            direction: t.direction || 'UNKNOWN'
          };
        });

        const buf = rtFrameBufferRef.current;
        buf.set(frameIdx, renderTracks);

        if (buf.size > RT_BUFFER_FRAMES) {
          const keys = Array.from(buf.keys()).sort((a, b) => a - b);
          for (let i = 0; i < keys.length - RT_BUFFER_FRAMES; i++) {
            buf.delete(keys[i]);
          }
        }

        // Phase 22.3: Decoupled Non-Blocking Playback
        // Maintain smooth 1.0x playback rate without jarring currentTime seeks
        if (videoRef.current && videoRef.current.readyState > 1) {
          videoRef.current.playbackRate = 1.0;
        }

        setLiveTrackCount(tracks.length);

        if (!firstBoxRenderedRef.current && tracks.length > 0 && uploadStartRef.current > 0) {
          firstBoxRenderedRef.current = true;
          const latencyMs = performance.now() - uploadStartRef.current;
          setFirstBoxMs(Math.round(latencyMs));
        }

      } else if (type === 'perception_metrics') {
        setPerceptionMetrics(payload);
      } else if (type === 'stream_status' || type === 'status' || msg.type === 'status') {
        const progress = msg.progress || 0;
        setJobProgress(progress);
        const stage = msg.stage || '';
        if (msg.video_fps) {
          setTelemetry((prev: any) => ({
            ...(prev || {}),
            video_fps: msg.video_fps,
            total_frames: msg.total_frames || prev?.total_frames
          }));
        }
        if (stage === 'DETECTION_AND_TRACKING' || stage === 'STREAMING') {
          setPipelineStage('LIVE_INFERENCE');
          const veh = msg.vehicles_tracked || 0;
          const done = msg.frames_done || 0;
          const total = msg.total_frames || 0;
          setJobStatusText(
            `Job #${jobId} · ● YOLOX Active · ${veh} vehicles · ${done}/${total} frames (${progress}%)`
          );
        } else if (stage === 'PERSISTING') {
          setPipelineStage('FINALIZING');
          setJobStatusText(`Job #${jobId} Finalizing — Saving trajectories to DB...`);
        } else if (stage === 'INITIALIZING') {
          setPipelineStage('PROCESSING');
          setJobStatusText(`Job #${jobId} Initializing CV Pipeline...`);
        }

      } else if (msg.type === 'completed') {
        isLiveStreamingRef.current = false;
        if (videoRef.current) {
          videoRef.current.playbackRate = 1.0;
          videoRef.current.loop = true;
          videoRef.current.play().catch(() => {});
        }
        setWsConnected(false);
        setPipelineStage('FINALIZING');
        setJobStatusText(`Job #${jobId} complete — loading finalized tracks...`);
        setProcessingFps(msg.processing_fps || null);
        setRealTimeFactor(msg.real_time_factor || null);
        if (msg.first_detection_latency_s && uploadStartRef.current > 0) {
          setFirstDetectionMs(Math.round(msg.first_detection_latency_s * 1000));
        }

        const jId = msg.job_id || activeJobIdRef.current;
        if (jId) {
          loadJobResults(jId);
        }

      } else if (msg.type === 'error') {
        isLiveStreamingRef.current = false;
        setWsConnected(false);
        setJobStatusText(`Error on Job #${jobId}: ${msg.message}`);
        setPipelineStage('ERROR');
      }
    };

    ws.onerror = () => {
      setWsConnected(false);
      // Immediately check REST API for job completion
      fetch(apiUrl(`/api/v1/jobs/${jobId}`)).then(r => r.json()).then(j => {
        if (j.status === 'SUCCEEDED') {
          isLiveStreamingRef.current = false;
          loadJobResults(jobId);
        }
      }).catch(() => {});
    };

    ws.onclose = () => {
      setWsConnected(false);
      // Resilient fallback: If WebSocket disconnects while job is still processing,
      // poll the REST endpoint every second until terminal state (SUCCEEDED/FAILED/CANCELLED).
      const pollTimer = setInterval(async () => {
        try {
          const res = await fetch(apiUrl(`/api/v1/jobs/${jobId}`));
          if (!res.ok) {
            clearInterval(pollTimer);
            return;
          }
          const j = await res.json();
          if (j.status === 'SUCCEEDED') {
            clearInterval(pollTimer);
            isLiveStreamingRef.current = false;
            setPipelineStage('FINALIZING');
            setJobStatusText(`Job #${jobId} complete — loading finalized tracks...`);
            loadJobResults(jobId);
          } else if (j.status === 'FAILED' || j.status === 'CANCELLED') {
            clearInterval(pollTimer);
            isLiveStreamingRef.current = false;
            setPipelineStage('ERROR');
            setJobStatusText(`Job #${jobId} ${j.status.toLowerCase()}: ${j.error_message || ''}`);
          } else if (j.status === 'RUNNING') {
            setJobProgress(Math.round(j.progress_pct || 0));
            setJobStatusText(`Job #${jobId} · ● YOLOX Active (${Math.round(j.progress_pct || 0)}%)`);
          }
        } catch {
          clearInterval(pollTimer);
        }
      }, 1000);
    };
  }, [loadJobResults]);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // =========================================================================
  // Playback & Frame Transport Controls
  // =========================================================================
  const totalFrames = isRealVideo
    ? (telemetry?.total_frames || (videoRef.current?.duration ? Math.floor(videoRef.current.duration * (telemetry?.video_fps || 29.97)) : N))
    : N;

  const handleTogglePlay = () => {
    const next = !playing;
    setPlaying(next);
    if (isRealVideoRef.current && videoRef.current) {
      if (next) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
      }
    }
  };

  const handleStepFrame = (delta: number) => {
    setPlaying(false);
    if (isRealVideoRef.current && videoRef.current) {
      videoRef.current.pause();
      const vFps = telemetry?.video_fps || 29.97;
      const targetTime = Math.max(0, (videoRef.current.currentTime || 0) + (delta / vFps));
      videoRef.current.currentTime = targetTime;
      setFrame(Math.floor(targetTime * vFps));
    } else {
      setFrame(f => Math.max(0, Math.min(N - 1, f + delta)));
    }
  };

  const handleSeek = (newFrame: number) => {
    setFrame(newFrame);
    seenRef.current.clear();
    if (isRealVideoRef.current && videoRef.current) {
      const vFps = telemetry?.video_fps || 29.97;
      videoRef.current.currentTime = Math.max(0, newFrame / vFps);
    }
  };

  const handleInspectTrack = (targetTrackId: number, preferredFrame?: number) => {
    setSelTrack(targetTrackId);
    setSidebarTab('inspector');
    setPlaying(false);

    let targetFrame: number | null = preferredFrame ?? null;

    if (targetFrame === null) {
      if (isRealVideoRef.current) {
        // 1. Check realTracks (finalized DB tracks)
        if (realTracks && realTracks.length > 0) {
          const trk = realTracks.find((t: any) => t.track_id === targetTrackId);
          if (trk) {
            const traj = Array.isArray(trk.trajectory)
              ? trk.trajectory
              : (typeof trk.trajectory === 'string' ? (JSON.parse(trk.trajectory) || []) : []);
            if (traj && traj.length > 0) {
              const midIdx = Math.floor(traj.length / 2);
              targetFrame = traj[midIdx]?.frame_index ?? traj[midIdx]?.frame ?? traj[0]?.frame_index ?? traj[0]?.frame ?? null;
            } else if (typeof trk.first_frame === 'number' && trk.first_frame >= 0) {
              targetFrame = trk.first_frame;
            }
          }
        }

        // 1b. Check examTracks (examination mode tracks)
        if (targetFrame === null && examTracks && examTracks.length > 0) {
          const eTrk = examTracks.find((t: any) => t.track_id === targetTrackId);
          if (eTrk) {
            const traj = eTrk.trajectory || [];
            if (traj.length > 0) {
              const midIdx = Math.floor(traj.length / 2);
              targetFrame = traj[midIdx]?.frame ?? traj[midIdx]?.frame_index ?? traj[0]?.frame ?? traj[0]?.frame_index ?? null;
            }
          }
        }

        // 1c. Check examEvents (examination events)
        if (targetFrame === null && examEvents && examEvents.length > 0) {
          const ev = examEvents.find((e: any) => e.person_track_id === targetTrackId);
          if (ev && typeof ev.start_frame === 'number') {
            targetFrame = ev.start_frame;
          }
        }

        // 2. Check frameMapRef (pre-computed frames from DB)
        if (targetFrame === null && frameMapRef.current && frameMapRef.current.size > 0) {
          const matchedFrames: number[] = [];
          for (const [fIndex, dets] of frameMapRef.current.entries()) {
            if (dets && dets.some((d: any) => d.track_id === targetTrackId)) {
              matchedFrames.push(fIndex);
            }
          }
          if (matchedFrames.length > 0) {
            targetFrame = matchedFrames[Math.floor(matchedFrames.length / 2)];
          }
        }

        // 3. Check rtFrameBufferRef (live streaming frame buffer)
        if (targetFrame === null && rtFrameBufferRef.current && rtFrameBufferRef.current.size > 0) {
          const matchedFrames: number[] = [];
          for (const [fIndex, dets] of rtFrameBufferRef.current.entries()) {
            if (dets && dets.some((d: any) => d.track_id === targetTrackId)) {
              matchedFrames.push(fIndex);
            }
          }
          if (matchedFrames.length > 0) {
            targetFrame = matchedFrames[Math.floor(matchedFrames.length / 2)];
          }
        }
      } else {
        // Synthetic demo vehicles
        const demoVeh = V.find(v => v.id === targetTrackId);
        if (demoVeh) {
          targetFrame = Math.max(0, Math.floor(((demoVeh.off || 0) / (demoVeh.v || 20)) * FPS) % N);
        }
      }
    }

    const vFps = telemetry?.video_fps || 29.97;
    if (targetFrame !== null && targetFrame >= 0) {
      setFrame(targetFrame);
      if (isRealVideoRef.current && videoRef.current) {
        videoRef.current.pause();
        const targetTime = targetFrame / vFps;
        videoRef.current.currentTime = Math.max(0, targetTime);
        setRenderTick(t => t + 1);
      } else {
        setRenderTick(t => t + 1);
      }
    } else {
      setRenderTick(t => t + 1);
    }

    // Scroll smoothly up to the video player canvas so the user immediately sees the vehicle in the clip
    setTimeout(() => {
      const container = canvasRef.current || videoRef.current;
      if (container) {
        container.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }, 40);
  };

  // =========================================================================
  // Synchronized Animation Loop for Video & Synthetic Demo
  // =========================================================================
  useEffect(() => {
    let animId: number;
    let lastTime = 0;
    const loop = (ts: number) => {
      if (playing) {
        if (isRealVideoRef.current && videoRef.current) {
          const vFps = telemetry?.video_fps || 29.97;
          const curF = Math.floor((videoRef.current.currentTime || 0) * vFps);
          if (curF !== lastFrameRef.current) {
            lastFrameRef.current = curF;
            setFrame(curF);
          }
        } else if (ts - lastTime > 1000 / FPS) {
          lastTime = ts;
          setFrame(f => { const next = (f + 1) % N; if (next === 0) seenRef.current.clear(); return next; });
        }
      }
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [playing, telemetry]);

  // =========================================================================
  // Main Canvas Render — 100% Exact Frame Matching & Scaling (Requirement #9, 11, 12)
  // =========================================================================
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const cx = cv.getContext('2d');
    if (!cx) return;

    const t = frame / FPS;

    if (isRealVideoRef.current && videoRef.current && (videoRef.current.readyState >= 1 || videoRef.current.videoWidth > 0)) {
      const vidElem = videoRef.current;

      // Source resolution (1920x1080 for Traffic1.mp4)
      const srcW = vidElem.videoWidth || telemetry?.source_width || 1920;
      const srcH = vidElem.videoHeight || telemetry?.source_height || 1080;
      const scaleX = W / srcW;
      const scaleY = H / srcH;

      const currentVidTime = vidElem.currentTime;
      const videoFps = telemetry?.video_fps || 29.97;
      const currentVidFrame = Math.floor(currentVidTime * videoFps);

      // Draw background video frame
      cx.drawImage(vidElem, 0, 0, W, H);

      // ---------------------------------------------------------------------
      // CONTEXTUAL OVERLAY: EXAMINATION SEATING ZONES vs TRAFFIC ROAD CALIBRATION
      // ---------------------------------------------------------------------
      if (activeModule === 'examination') {
        // Seating Zones Overlay
        if (examZones.length > 0) {
          examZones.forEach(z => {
            if (!z.polygon || z.polygon.length < 3) return;
            const pts = z.polygon.map(([x, y]: [number, number]) => [x * scaleX, y * scaleY]);
            const isRestricted = z.zone_type.includes('RESTRICTED');
            cx.save();
            cx.fillStyle = isRestricted ? 'rgba(239, 68, 68, 0.12)' : 'rgba(99, 102, 241, 0.10)';
            cx.strokeStyle = isRestricted ? '#EF4444' : '#6366F1';
            cx.lineWidth = 1.5;
            cx.setLineDash([4, 2]);
            cx.beginPath();
            pts.forEach((p: number[], i: number) => (i ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])));
            cx.closePath();
            cx.fill();
            cx.stroke();

            // Zone tag
            cx.fillStyle = isRestricted ? '#EF4444' : '#818CF8';
            cx.font = 'bold 9px sans-serif';
            cx.textAlign = 'left';
            cx.fillText(`📍 ${z.zone_name} (${z.zone_type})`, pts[0][0] + 4, pts[0][1] + 12);
            cx.restore();
          });
        }

        // Examination Contextual Status Banner
        const bannerW = 460;
        const bannerH = 20;
        const bannerX = 12;
        const bannerY = 10;
        cx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        cx.strokeStyle = '#6366F1';
        cx.lineWidth = 1;
        cx.beginPath();
        if ((cx as any).roundRect) (cx as any).roundRect(bannerX, bannerY, bannerW, bannerH, 4);
        else cx.rect(bannerX, bannerY, bannerW, bannerH);
        cx.fill(); cx.stroke();

        cx.fillStyle = '#A5B4FC';
        cx.font = 'bold 9.5px sans-serif';
        cx.textAlign = 'left';
        cx.fillText('🎓 EXAMINATION INTELLIGENCE · Person & Object Tracking · Zone Association · Event Proctorship', bannerX + 8, bannerY + 14);

      } else {
        // Traffic Calibration Overlay
        if (realCalib?.image_points_json && realCalib.image_points_json.length >= 4) {
          const calibPts = realCalib.image_points_json;
          const displayPts = calibPts.map(([x, y]: [number, number]) => [x * scaleX, y * scaleY]);

          cx.strokeStyle = '#3B82F6'; cx.fillStyle = '#3B82F61A'; cx.lineWidth = 1.5;
          cx.beginPath();
          displayPts.forEach((p: number[], i: number) => (i ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])));
          cx.closePath(); cx.fill(); cx.stroke();

          displayPts.forEach((p: number[], i: number) => {
            cx.fillStyle = '#2563EB'; cx.beginPath(); cx.arc(p[0], p[1], 4, 0, 2 * Math.PI); cx.fill();
            cx.fillStyle = '#FFFFFF'; cx.font = 'bold 9px sans-serif'; cx.textAlign = 'center';
            cx.fillText('P' + (i + 1), p[0], p[1] + 3);
          });

          // Label on road polygon
          if (displayPts.length >= 4) {
            cx.fillStyle = '#2563EBEE'; cx.font = '600 10px sans-serif'; cx.textAlign = 'center';
            cx.fillText('CALIBRATED ROAD REGION', (displayPts[0][0] + displayPts[1][0]) / 2, displayPts[0][1] + 16);
          }
        }

        const hasCalib = Boolean(realCalib?.image_points_json && realCalib.image_points_json.length >= 4);
        const bannerW = hasCalib ? 380 : 440;
        const bannerH = 20;
        const bannerX = 12;
        const bannerY = 10;
        cx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        cx.strokeStyle = hasCalib ? '#10B981' : '#64748B';
        cx.lineWidth = 1;
        cx.beginPath();
        if ((cx as any).roundRect) (cx as any).roundRect(bannerX, bannerY, bannerW, bannerH, 4);
        else cx.rect(bannerX, bannerY, bannerW, bannerH);
        cx.fill(); cx.stroke();

        cx.fillStyle = hasCalib ? '#34D399' : '#CBD5E1';
        cx.font = 'bold 9.5px sans-serif';
        cx.textAlign = 'left';
        const bannerText = hasCalib
          ? '✓ ROAD CALIBRATION ACTIVE · Monocular metric speed estimation enabled'
          : '⚠️ UNCALIBRATED CAMERA · Speed requires road geometry calibration · Tracking active';
        cx.fillText(bannerText, bannerX + 8, bannerY + 14);
      }

      // ---------------------------------------------------------------------
      // REAL Bounding Box Rendering — Frame Sync Contract (Smooth Multi-Frame Matching)
      // ---------------------------------------------------------------------
      const findDets = (f: number, map: Map<number, any[]>, tol: number = 4) => {
        if (map.has(f)) return map.get(f);
        for (let i = 1; i <= tol; i++) {
          if (map.has(f - i)) return map.get(f - i);
          if (map.has(f + i)) return map.get(f + i);
        }
        return undefined;
      };

      const getInterpolatedDets = (exactFrame: number, map: Map<number, any[]>, tol: number = 4) => {
        const f0 = Math.floor(exactFrame);
        const f1 = Math.ceil(exactFrame);
        const alpha = exactFrame - f0;

        const baseDets = map.get(f0) || findDets(f0, map, tol);
        if (!baseDets || baseDets.length === 0) return baseDets;
        if (alpha < 0.001 || f0 === f1) return baseDets;

        const nextDets = map.get(f1);
        if (!nextDets || nextDets.length === 0) return baseDets;

        const nextMap = new Map<number, any>();
        nextDets.forEach(d => nextMap.set(d.track_id, d));

        return baseDets.map(d0 => {
          const d1 = nextMap.get(d0.track_id);
          if (!d1) return d0;
          return {
            ...d0,
            rx: d0.rx + alpha * (d1.rx - d0.rx),
            ry: d0.ry + alpha * (d1.ry - d0.ry),
            rw: d0.rw + alpha * (d1.rw - d0.rw),
            rh: d0.rh + alpha * (d1.rh - d0.rh),
          };
        });
      };

      let currentFrameDets: any[] | undefined;
      if (isLiveStreamingRef.current || frameMapRef.current.size === 0) {
        currentFrameDets = findDets(currentVidFrame, rtFrameBufferRef.current, 5);

        if (!currentFrameDets && latestInferenceFrame !== null && latestInferenceFrame !== undefined) {
          const allKeys = Array.from(rtFrameBufferRef.current.keys());
          if (allKeys.length > 0) {
            const closest = allKeys.reduce((prev, curr) => Math.abs(curr - currentVidFrame) < Math.abs(prev - currentVidFrame) ? curr : prev);
            // Strict bounded temporal matching: NEVER render stale detections if drift exceeds 30 frames (~1.0s)
            if (Math.abs(closest - currentVidFrame) <= 30) {
              currentFrameDets = rtFrameBufferRef.current.get(closest);
            }
          }
        }
      }
      if (!currentFrameDets && frameMapRef.current.size > 0) {
        currentFrameDets = getInterpolatedDets(currentVidTime * videoFps, frameMapRef.current, 5);
      }

      if (currentFrameDets && currentFrameDets.length > 0) {
        // --- 1. Filter tracks per user controls while strictly preserving selected track ---
        let visibleDets = currentFrameDets;
        if (focusSelected && selTrack !== null) {
          visibleDets = visibleDets.filter(d => d.track_id === selTrack);
        } else {
          if (activeModule !== 'examination') {
            if (classFilter !== 'ALL') {
              visibleDets = visibleDets.filter(d => d.track_id === selTrack || (d.vehicle_class || 'car').toUpperCase() === classFilter);
            }
          }
          if (minConfidence > 0) {
            const effectiveMinConf = activeModule === 'examination' ? Math.min(minConfidence, 0.18) : minConfidence;
            visibleDets = visibleDets.filter(d => d.track_id === selTrack || (d.confidence ?? 0.9) >= effectiveMinConf);
          }
        }

        // --- 2. Trajectories: Draw historical trails if enabled or in INTELLIGENCE mode ---
        if (showTrajectories || vizMode === 'INTELLIGENCE') {
          visibleDets.forEach(det => {
            const isSel = selTrack === det.track_id;
            const isHov = hoveredTrack === det.track_id;
            if (showTrajectories || isSel || isHov) {
              const trkId = det.track_id;
              const points: [number, number][] = [];
              if (isLiveStreamingRef.current) {
                for (let f = Math.max(0, currentVidFrame - 40); f <= currentVidFrame; f += 2) {
                  const fDets = rtFrameBufferRef.current.get(f);
                  const inst = fDets?.find((d: any) => d.track_id === trkId);
                  if (inst) {
                    points.push([(inst.rx + inst.rw / 2) * scaleX, (inst.ry + inst.rh / 2) * scaleY]);
                  }
                }
              } else {
                const trk = realTracks.find((t: any) => t.track_id === trkId);
                if (trk?.trajectory && Array.isArray(trk.trajectory)) {
                  trk.trajectory
                    .filter((pt: any) => pt.frame <= currentVidFrame && pt.frame >= currentVidFrame - 45)
                    .forEach((pt: any) => {
                      const px = (pt.center_x ?? (pt.x + pt.w / 2)) * scaleX;
                      const py = (pt.center_y ?? (pt.y + pt.h / 2)) * scaleY;
                      points.push([px, py]);
                    });
                }
              }

              if (points.length >= 2) {
                cx.save();
                cx.strokeStyle = isSel ? '#F59E0B' : isHov ? '#38BDF8' : '#3B82F688';
                cx.lineWidth = isSel ? 2.5 : 1.5;
                cx.setLineDash(isSel ? [] : [3, 2]);
                cx.beginPath();
                points.forEach(([px, py], idx) => {
                  if (idx === 0) cx.moveTo(px, py);
                  else cx.lineTo(px, py);
                });
                cx.stroke();
                cx.restore();
              }
            }
          });
        }

        // --- 3. Bounding Boxes & Intelligent Badges ---
        const renderedBadges: { x: number, y: number, w: number, h: number }[] = [];
        let labelCount = 0;

        visibleDets.forEach(det => {
          const bx = Math.min(det.rx, det.rx + det.rw) * scaleX;
          const by = Math.min(det.ry, det.ry + det.rh) * scaleY;
          const bw = Math.max(18, Math.abs(det.rw) * scaleX);
          const bh = Math.max(16, Math.abs(det.rh) * scaleY);

          const isCalibratedSpeed = det.measurementState === 'CALIBRATED_SPEED' ||
            (det.speedKmhCalibrated !== null && det.speedKmhCalibrated !== undefined && det.speedKmhCalibrated > 0);
          const hasValidSpeed = isCalibratedSpeed;
          const isUncalibratedMotion = det.measurementState === 'UNCALIBRATED_MOTION' ||
            det.speedStatus === 'UNCALIBRATED_MOTION' || det.speedStatus === 'UNCALIBRATED_ESTIMATE';

          const isCalculating = !hasValidSpeed && !isUncalibratedMotion && (
            det.speedStatus === 'INSUFFICIENT_DATA' ||
            det.speedStatus === 'INSUFFICIENT_HISTORY' ||
            det.speedStatus === 'LOW_CONFIDENCE' ||
            det.speedStatus === 'CALCULATING...'
          );
          const isCalibReq = !hasValidSpeed && !isUncalibratedMotion && det.speedStatus === 'CALIBRATION_REQUIRED';

          // SPEEDING INTEGRITY: uncalibrated motion cannot trigger a confirmed speeding violation
          const bad = isCalibratedSpeed && det.speedKmh > speedLimit;
          const isSelected = selTrack === det.track_id;
          const isHovered = hoveredTrack === det.track_id;

          let col = '#94A3B8';
          if (activeModule === 'examination') {
            const isStaff = ['INVIGILATOR', 'EXAM_COORDINATOR', 'AUTHORIZED_STAFF'].includes(det.assigned_role || '');
            const hasActiveRedEvent = (det.active_events && det.active_events.some((e: any) => e.priority === 'RED')) ||
              examEvents.some(e => e.person_track_id === det.track_id && e.priority === 'RED' && e.review_status !== 'REJECTED');
            const hasActiveAmberEvent = (det.active_events && det.active_events.some((e: any) => e.priority === 'AMBER')) ||
              examEvents.some(e => e.person_track_id === det.track_id && e.priority === 'AMBER' && e.review_status !== 'REJECTED');

            col = isSelected ? '#F59E0B'
              : isHovered ? '#38BDF8'
              : hasActiveRedEvent ? '#EF4444'
              : hasActiveAmberEvent ? '#F59E0B'
              : isStaff ? '#06B6D4'
              : det.entity_type === 'person' ? (det.assigned_role === 'STUDENT' ? '#3B82F6' : '#94A3B8')
              : '#A855F7';
          } else {
            col = isSelected ? '#F59E0B'
              : isHovered ? '#38BDF8'
              : bad ? '#EF4444'
              : isCalibratedSpeed ? '#10B981'
              : isUncalibratedMotion ? '#0EA5E9'
              : isCalibReq ? '#64748B'
              : isCalculating ? '#F59E0B'
              : '#94A3B8';
          }

          // Box interior
          cx.fillStyle = isSelected ? 'rgba(245, 158, 11, 0.22)' : isHovered ? 'rgba(56, 189, 248, 0.20)' : (col + '20');
          cx.fillRect(bx, by, bw, bh);

          // Box brackets
          const bracketLen = Math.min(12, Math.min(bw, bh) * 0.35);
          cx.strokeStyle = col;
          cx.lineWidth = (isSelected || isHovered) ? 2.5 : 1.5;
          cx.beginPath();
          cx.moveTo(bx, by + bracketLen); cx.lineTo(bx, by); cx.lineTo(bx + bracketLen, by);
          cx.moveTo(bx + bw - bracketLen, by); cx.lineTo(bx + bw, by); cx.lineTo(bx + bw, by + bracketLen);
          cx.moveTo(bx + bw, by + bh - bracketLen); cx.lineTo(bx + bw, by + bh); cx.lineTo(bx + bw - bracketLen, by + bh);
          cx.moveTo(bx + bracketLen, by + bh); cx.lineTo(bx, by + bh); cx.lineTo(bx, by + bh - bracketLen);
          cx.stroke();

          // Subtle box border
          cx.strokeStyle = col + '33';
          cx.lineWidth = 1;
          cx.strokeRect(bx, by, bw, bh);

          // Center target dot
          cx.fillStyle = col;
          cx.beginPath();
          cx.arc(bx + bw / 2, by + bh / 2, (isSelected || isHovered) ? 3 : 2, 0, 2 * Math.PI);
          cx.fill();

          // --- 4. Intelligent Label Generation & Decluttering ---
          // Rule: Selected or Hovered ALWAYS render detailed label.
          // In MINIMAL mode: non-selected/non-hovered have NO labels.
          // If labels toggled off: non-selected/non-hovered have NO labels.
          // Density cap: maxLabels limits simultaneously rendered badges.
          const shouldRenderBadge = isSelected || isHovered || (showLabels && vizMode !== 'MINIMAL' && labelCount < maxLabels);
          if (!shouldRenderBadge) return;

          let tx = '';

          if (activeModule === 'examination') {
            if (det.entity_type === 'person') {
              const conf = Math.round((det.confidence || 0.9) * 100);
              const role = det.assigned_role || 'UNKNOWN';
              const rolePrefix = (role === 'INVIGILATOR' || role === 'AUTHORIZED_STAFF' || role === 'EXAM_COORDINATOR') ? 'STAFF ' : (role === 'STUDENT' ? 'STU ' : '');
              const hasRed = examEvents.some(e => e.person_track_id === det.track_id && e.priority === 'RED' && e.review_status !== 'REJECTED');
              const alertPrefix = hasRed ? '🚨 ' : '';
              const zoneStr = det.assigned_zone ? ` · [${det.assigned_zone}]` : '';
              tx = `${alertPrefix}${rolePrefix}#${det.track_id} · ${conf}%${zoneStr}`;
            } else {
              const conf = Math.round((det.confidence || 0.8) * 100);
              const isUncertain = det.entity_type === 'UNCERTAIN_ELECTRONIC_DEVICE';
              const icon = isUncertain ? '❓📱 ' : (det.entity_type === 'cell phone' ? '📱 ' : '📖 ');
              tx = `${icon}${(det.entity_type || 'OBJECT').toUpperCase()} · ${conf}%`;
            }
          } else {
            const clsLabel = (det.vehicle_class || 'Vehicle').toUpperCase();

            if (isSelected || isHovered) {
              const conf = Math.round((det.confidence || 0.9) * 100);
              if (isCalibratedSpeed) {
                tx = `${clsLabel} #${det.track_id} · ${conf}% · ${Math.round(det.speedKmh)} km/h`;
              } else if (isUncalibratedMotion) {
                tx = `${clsLabel} #${det.track_id} · ${conf}% · [UNCALIB]`;
              } else {
                tx = `${clsLabel} #${det.track_id} · ${conf}%`;
              }
            } else if (vizMode === 'STANDARD') {
              if (isCalibratedSpeed) {
                tx = `${clsLabel} #${det.track_id} · ${Math.round(det.speedKmh)} km/h`;
              } else {
                tx = `${clsLabel} #${det.track_id}`;
              }
            } else if (vizMode === 'INTELLIGENCE') {
              const conf = Math.round((det.confidence || 0.9) * 100);
              if (isCalibratedSpeed) {
                tx = `${clsLabel} #${det.track_id} · ${conf}% · ${Math.round(det.speedKmh)} km/h`;
              } else if (isUncalibratedMotion) {
                tx = `${clsLabel} #${det.track_id} · ${conf}% · [Motion]`;
              } else {
                tx = `${clsLabel} #${det.track_id} · ${conf}%`;
              }
            }
          }

          cx.font = (isSelected || isHovered) ? 'bold 11px monospace, sans-serif' : '10px monospace, sans-serif';
          const tw = cx.measureText(tx).width + 12;
          const badgeH = 18;
          const badgeX = Math.max(2, Math.min(W - tw - 2, bx + bw / 2 - tw / 2));
          let badgeY = by >= 22 ? by - 20 : by + bh + 4;

          // Deterministic collision detection with already-rendered badges
          let collides = false;
          for (const prev of renderedBadges) {
            const overlapX = Math.abs(badgeX - prev.x) < Math.max(tw, prev.w) * 0.88;
            const overlapY = Math.abs(badgeY - prev.y) < (badgeH - 2);
            if (overlapX && overlapY) {
              collides = true;
              break;
            }
          }

          if (collides) {
            const altY = (by >= 22 && badgeY === by - 20) ? (by + bh + 4) : (by >= 22 ? by - 20 : badgeY);
            let altCollides = false;
            for (const prev of renderedBadges) {
              const overlapX = Math.abs(badgeX - prev.x) < Math.max(tw, prev.w) * 0.88;
              const overlapY = Math.abs(altY - prev.y) < (badgeH - 2);
              if (overlapX && overlapY) {
                altCollides = true;
                break;
              }
            }
            if (!altCollides && altY >= 2 && altY + badgeH <= H - 2) {
              badgeY = altY;
              collides = false;
            }
          }

          // If still colliding and NOT selected/hovered: suppress badge to maintain legibility
          if (collides && !isSelected && !isHovered) {
            return;
          }

          badgeY = Math.max(2, Math.min(H - badgeH - 2, badgeY));
          renderedBadges.push({ x: badgeX, y: badgeY, w: tw, h: badgeH });
          labelCount++;

          // Draw Badge Background & Border
          cx.fillStyle = '#0B0F19F2';
          cx.strokeStyle = col;
          cx.lineWidth = (isSelected || isHovered) ? 1.8 : 1.2;
          cx.beginPath();
          if ((cx as any).roundRect) (cx as any).roundRect(badgeX, badgeY, tw, badgeH, 4);
          else cx.rect(badgeX, badgeY, tw, badgeH);
          cx.fill(); cx.stroke();

          cx.fillStyle = isSelected ? '#FBBF24'
            : isHovered ? '#7DD3FC'
            : bad ? '#FCA5A5'
            : isCalibratedSpeed ? '#6EE7B7'
            : isUncalibratedMotion ? '#7DD3FC'
            : '#CBD5E1';
          cx.textAlign = 'center';
          cx.fillText(tx, badgeX + tw / 2, badgeY + 13);
        });

        // --- 5. Selected / Hovered Floating Object HUD Card ---
        const activeInspectorId = hoveredTrack ?? selTrack;
        if (activeInspectorId !== null) {
          const inspectedDet = currentFrameDets.find(d => d.track_id === activeInspectorId);
          if (inspectedDet) {
            const cardX = 12;
            const cardY = H - 54;
            const cardW = 340;
            const cardH = 44;
            cx.fillStyle = 'rgba(15, 23, 42, 0.92)';
            cx.strokeStyle = hoveredTrack === activeInspectorId ? '#38BDF8' : '#F59E0B';
            cx.lineWidth = 1.5;
            cx.beginPath();
            if ((cx as any).roundRect) (cx as any).roundRect(cardX, cardY, cardW, cardH, 6);
            else cx.rect(cardX, cardY, cardW, cardH);
            cx.fill(); cx.stroke();

            const cls = (inspectedDet.vehicle_class || 'Vehicle').toUpperCase();
            const spdText = (inspectedDet.speedKmh && inspectedDet.speedKmh > 0)
              ? `${Math.round(inspectedDet.speedKmh)} km/h`
              : inspectedDet.measurementState === 'UNCALIBRATED_MOTION'
                ? 'Uncalibrated motion'
                : 'Calibration required';
            const stateColor = (inspectedDet.speedKmh && inspectedDet.speedKmh > 0)
              ? '#10B981'
              : inspectedDet.measurementState === 'UNCALIBRATED_MOTION'
                ? '#0EA5E9'
                : '#F59E0B';

            cx.font = 'bold 11px sans-serif';
            cx.fillStyle = hoveredTrack === activeInspectorId ? '#38BDF8' : '#F59E0B';
            cx.textAlign = 'left';
            cx.fillText(`TRACK #${activeInspectorId} [${cls}]`, cardX + 10, cardY + 17);

            cx.font = '10px monospace, sans-serif';
            cx.fillStyle = '#94A3B8';
            cx.fillText(`Box: [${Math.round(inspectedDet.rx)}, ${Math.round(inspectedDet.ry)}, ${Math.round(inspectedDet.rw)}x${Math.round(inspectedDet.rh)}] · Conf: ${Math.round((inspectedDet.confidence || 0.9) * 100)}%`, cardX + 10, cardY + 34);

            cx.font = 'bold 11px sans-serif';
            cx.fillStyle = stateColor;
            cx.textAlign = 'right';
            cx.fillText(spdText, cardX + cardW - 10, cardY + 22);
          }
        }
      }

      // LIVE indicator badge (top right corner) — Live AI Telemetry HUD
      if (isLiveStreamingRef.current) {
        const infFrameText = latestInferenceFrame !== null && latestInferenceFrame !== undefined ? `F#${latestInferenceFrame}` : 'F#0';
        const fpsText = processingFps ? `${processingFps.toFixed(1)} FPS` : 'LIVE';
        const badge = `● REAL VIDEO — LIVE AI | ${infFrameText} (${fpsText}) | JOB #${activeJobId || 'NONE'}`;
        cx.font = 'bold 10px monospace, sans-serif';
        const bw2 = cx.measureText(badge).width + 16;
        cx.fillStyle = '#0F172AEE'; cx.strokeStyle = '#10B981'; cx.lineWidth = 1.5;
        cx.beginPath();
        if ((cx as any).roundRect) (cx as any).roundRect(W - bw2 - 6, 8, bw2, 20, 5);
        else cx.rect(W - bw2 - 6, 8, bw2, 20);
        cx.fill(); cx.stroke();
        cx.fillStyle = '#10B981'; cx.textAlign = 'right';
        cx.fillText(badge, W - 14, 22);
      }

    } else if (isRealVideoRef.current) {
      // Real video mode but video player loading
      cx.fillStyle = '#0F172A';
      cx.fillRect(0, 0, W, H);
      cx.fillStyle = '#4F46E5';
      cx.font = 'bold 18px sans-serif';
      cx.textAlign = 'center';
      cx.fillText('Loading Real Traffic Video...', W / 2, H / 2 - 10);
      cx.fillStyle = '#64748B'; cx.font = '12px sans-serif';
      cx.fillText('Select or upload a real video file to begin YOLOX + ByteTrack analysis', W / 2, H / 2 + 18);

    } else {
      // =====================================================================
      // DEMO MODE — SYNTHETIC ONLY (Requirement #7)
      // =====================================================================
      const g = cx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, '#BAE6FD'); g.addColorStop(0.2, '#E2E8F0');
      cx.fillStyle = g; cx.fillRect(0, 0, W, H);
      cx.fillStyle = '#94A3B8'; cx.beginPath();
      [[-1, 150], [13, 150], [13, 0.5], [-1, 0.5]].forEach(([a, b], i) => {
        const p = proj(a, b); i ? cx.lineTo(...p) : cx.moveTo(...p);
      }); cx.fill();
      cx.strokeStyle = '#F8FAFC'; cx.lineWidth = 2; cx.setLineDash([14, 12]);
      for (const lx of [4, 8]) { cx.beginPath(); cx.moveTo(...proj(lx, 150)); cx.lineTo(...proj(lx, 0.5)); cx.stroke(); }
      cx.setLineDash([]);
      cx.strokeStyle = '#4F46E5'; cx.fillStyle = '#4F46E522'; cx.lineWidth = 2; cx.beginPath();
      HP.forEach((p, i) => (i ? cx.lineTo(...p) : cx.moveTo(...p)));
      cx.closePath(); cx.fill(); cx.stroke();
      HP.forEach((p, i) => { cx.fillStyle = '#4F46E5'; cx.beginPath(); cx.arc(p[0], p[1], 7, 0, 7); cx.fill(); cx.fillStyle = '#fff'; cx.font = 'bold 9px sans-serif'; cx.textAlign = 'center'; cx.fillText('P' + (i + 1), p[0], p[1] + 3); });

      V.map(a => ({ a, s: state(a, t) })).filter(o => o.s.d > 6 && o.s.d < 125).sort((p, q) => q.s.d - p.s.d).forEach(({ a, s }) => {
        const v = estSpeed(a, frame); a.cur = v; const bad = v > speedLimit;
        const col = bad ? '#EF4444' : '#10B981'; const x = s.x - s.bw / 2; const y = s.y - s.bh;
        cx.fillStyle = { Truck: '#475569', SUV: '#1E293B', Sedan: '#64748B', Motorcycle: '#7C3AED' }[a.cls as string] || '#64748B';
        cx.fillRect(x, y, s.bw, s.bh);
        if (showTrajectories || vizMode === 'INTELLIGENCE') {
          cx.strokeStyle = '#4F46E5'; cx.lineWidth = 1; cx.setLineDash([4, 3]);
          cx.beginPath(); for (let k = 0; k <= 14; k++) { const q = state(a, (frame - k * 3) / FPS); k ? cx.lineTo(q.x, q.y) : cx.moveTo(q.x, q.y); } cx.stroke(); cx.setLineDash([]);
        }
        cx.strokeStyle = selTrack === a.id ? '#F59E0B' : col; cx.lineWidth = selTrack === a.id ? 3 : 2;
        cx.strokeRect(x - 2, y - 2, s.bw + 4, s.bh + 4);
        if (showLabels && (vizMode !== 'MINIMAL' || selTrack === a.id || hoveredTrack === a.id)) {
          const tx = Math.round(v) + ' km/h'; const fs = Math.max(9, 11 + s.s * 6); cx.font = 'bold ' + fs + 'px sans-serif';
          const tw = cx.measureText(tx).width + 10; cx.fillStyle = col; cx.beginPath();
          if ((cx as any).roundRect) (cx as any).roundRect(s.x - tw / 2, y - fs - 10, tw, fs + 5, 9); else cx.rect(s.x - tw / 2, y - fs - 10, tw, fs + 5);
          cx.fill(); cx.fillStyle = '#fff'; cx.textAlign = 'center'; cx.fillText(tx, s.x, y - 6);
        }
        if (bad && !seenRef.current.has(a.id) && s.d < 70) { seenRef.current.add(a.id); setViolations(prev => [{ t, id: a.id, v, loc: 'Lane ' + ((a.lx / 4 + 0.5) | 0) + ' · km 12.4' }, ...prev.slice(0, 29)]); }
      });

      const demoBadge = 'DEMO MODE — SYNTHETIC';
      cx.font = 'bold 10px sans-serif';
      const dbw = cx.measureText(demoBadge).width + 16;
      cx.fillStyle = '#0F172ACC'; cx.strokeStyle = '#F59E0B'; cx.lineWidth = 1.5;
      cx.beginPath();
      if ((cx as any).roundRect) (cx as any).roundRect(W - dbw - 6, 8, dbw, 20, 5); else cx.rect(W - dbw - 6, 8, dbw, 20);
      cx.fill(); cx.stroke();
      cx.fillStyle = '#FCD34D'; cx.textAlign = 'right'; cx.fillText(demoBadge, W - 14, 22);
    }
  }, [
    frame, renderTick, speedLimit, selTrack, hoveredTrack, isRealVideo, realTracks, telemetry,
    realCalib, latestInferenceFrame, processingFps, activeJobId,
    vizMode, showLabels, showTrajectories, classFilter, minConfidence,
    focusSelected, maxLabels, activeModule, examZones, examEvents
  ]);

  const startJobForModule = useCallback(async (videoId: number, chosenModule: AnalysisModule, preserveBanner: boolean = false) => {
    setActiveModule(chosenModule);
    if (!preserveBanner) setSceneRec(null);
    setPipelineStage('JOB_CREATED');
    setJobStatusText(`Initializing ${chosenModule.toUpperCase()} Intelligence Pipeline...`);

    let assignedCalib: any = null;
    if (chosenModule === 'traffic') {
      try {
        const calibRes = await fetch(apiUrl(`/api/v1/videos/${videoId}/calibrations`));
        if (calibRes.ok) {
          const calibData = await calibRes.json();
          if (calibData?.length) assignedCalib = calibData[0];
        }
      } catch (ce) {
        console.warn('Calibration query error:', ce);
      }
    }
    setRealCalib(assignedCalib);

    if (chosenModule === 'examination') {
      try {
        const zonesRes = await fetch(apiUrl(`/api/v1/videos/${videoId}/examination/zones`));
        if (zonesRes.ok) {
          const zData = await zonesRes.json();
          setExamZones(zData || []);
        }
      } catch (ze) {
        console.warn('Zones query error:', ze);
      }
    }

    try {
      const jobRes = await fetch(apiUrl(`/api/v1/videos/${videoId}/jobs`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          calibration_id: assignedCalib ? assignedCalib.id : null,
          config_json: { module: chosenModule, processing_mode: processingMode }
        })
      });
      if (!jobRes.ok) throw new Error(`Job creation failed: ${jobRes.status}`);
      const jobData = await jobRes.json();
      const jobId = jobData.id;

      setActiveJobId(jobId);
      activeJobIdRef.current = jobId;
      setPipelineStage('JOB_CREATED');
      setJobStatusText(`Job #${jobId} [${chosenModule.toUpperCase()}] running — connecting real-time stream...`);

      connectWebSocket(jobId);
    } catch (err: any) {
      console.error('Job error:', err);
      setJobStatusText(`Error: ${err.message}`);
      setPipelineStage('ERROR');
    }
  }, [connectWebSocket]);

  const handleReviewEvent = async (eventId: string, status: 'ACCEPTED' | 'REJECTED', notes: string) => {
    try {
      const res = await fetch(apiUrl(`/api/v1/examination/events/${eventId}/review`), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_status: status, reviewer_notes: notes })
      });
      if (res.ok) {
        setExamEvents(prev => prev.map(e => e.event_id === eventId ? { ...e, review_status: status, reviewer_notes: notes } : e));
      }
    } catch (e) {
      console.error('Review event error:', e);
    }
  };

  // =========================================================================
  // Save Road Calibration directly from Workbench
  // =========================================================================
  const handleSaveWorkbenchCalibration = async () => {
    const targetVideoId = activeVideoId || telemetry?.video_id;
    if (!targetVideoId) {
      alert('Please upload or select an active video first.');
      return;
    }
    try {
      const srcW = telemetry?.source_width || 1920;
      const srcH = telemetry?.source_height || 1080;
      const scaleX = srcW / W;
      const scaleY = srcH / H;
      const scaledHP = HP.map(([x, y]) => [Math.round(x * scaleX), Math.round(y * scaleY)]);

      const res = await fetch(apiUrl(`/api/v1/videos/${targetVideoId}/calibrations`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_points: scaledHP,
          world_points: [[0, 0], [12, 0], [12, 80], [0, 80]],
          camera_height_m: 8.5,
          pitch_deg: 14.0
        })
      });
      if (res.ok) {
        const calibData = await res.json();
        setRealCalib(calibData);
        alert(`✓ Road geometry calibration saved successfully (Version ${calibData.version})! Metric speed estimation is now active.`);
      } else {
        const err = await res.json();
        alert(`Calibration error: ${err.detail || 'Could not save calibration'}`);
      }
    } catch (e: any) {
      alert(`Network error saving calibration: ${e.message}`);
    }
  };

  // =========================================================================
  // Handle File Upload
  // =========================================================================
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    uploadStartRef.current = performance.now();
    firstBoxRenderedRef.current = false;

    setSrcName(file.name);
    setIsRealVideo(true);
    setRealTracks([]);
    setActiveJobId(null);
    setTelemetry(null);
    setRealCalib(null); // Clear stale calibration from previous video
    setSelTrack(null);
    setSelectedPersonId(null);
    setViolations([]);
    setExamTracks([]);
    setExamEvents([]);
    setExamZones([]);
    setLiveTrackCount(0);
    setLatestInferenceFrame(null);
    setFirstDetectionMs(null);
    setFirstBoxMs(null);
    setProcessingFps(null);
    setRealTimeFactor(null);
    rtFrameBufferRef.current.clear();
    frameMapRef.current.clear();
    isLiveStreamingRef.current = false;
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }

    const blobUrl = URL.createObjectURL(file);
    setVideoSrc(blobUrl);
    if (videoRef.current) {
      videoRef.current.src = blobUrl;
      videoRef.current.muted = true;
      videoRef.current.loop = false;
      videoRef.current.play().catch(() => {});
    }
    setPipelineStage('VIDEO_READY');
    setJobStatusText('Video loaded — Uploading for AI analysis...');

    setPipelineStage('UPLOADING');
    if (activeJobIdRef.current) {
      fetch(apiUrl(`/api/v1/jobs/${activeJobIdRef.current}/cancel`), { method: 'POST' }).catch(() => {});
    }
    rtFrameBufferRef.current.clear();
    frameMapRef.current.clear();
    seenRef.current.clear();
    lastRenderedFrameIdRef.current = -1;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const vidRes = await fetch(apiUrl('/api/v1/projects/1/videos'), { method: 'POST', body: formData });
      if (!vidRes.ok) throw new Error(`Upload failed: ${vidRes.status}`);
      const vidData = await vidRes.json();
      setPipelineStage('UPLOAD_COMPLETE');
      setJobStatusText('Video uploaded — Analyzing scene category with Scene Router...');
      setIsClassifying(true);

      // Authoritative extracted video metadata from backend
      setTelemetry({
        source_width: vidData.width || 1920,
        source_height: vidData.height || 1080,
        video_fps: vidData.fps || 29.97,
        video_duration_sec: vidData.duration_sec || 0.0
      });

      // Call Scene Router
      let rec: any = null;
      try {
        const classRes = await fetch(apiUrl(`/api/v1/videos/${vidData.id}/classify`), { method: 'POST' });
        if (classRes.ok) {
          rec = await classRes.json();
          setSceneRec({
            ...rec,
            video_id: vidData.id,
            video_name: file.name
          });
          setJobStatusText(`Scene estimated: ${rec.scene_type} (${Math.round(rec.confidence * 100)}%) — Auto-running ${rec.recommended_module.toUpperCase()} pipeline...`);
        }
      } catch (ce) {
        console.warn('Scene classification query error:', ce);
      }
      setIsClassifying(false);

      const targetModule: AnalysisModule = rec?.recommended_module || 'traffic';
      startJobForModule(vidData.id, targetModule, true);

    } catch (err: any) {
      console.error('Upload/job error:', err);
      setJobStatusText(`Error: ${err.message}`);
      setPipelineStage('ERROR');
    }
  };


  // =========================================================================
  // Canvas Mouse and Click Handlers (Requirement #2: Hover & Selection)
  // =========================================================================
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const r = cv.getBoundingClientRect();
    const x = ((e.clientX - r.left) * W) / r.width;
    const y = ((e.clientY - r.top) * H) / r.height;

    let hitTrk: number | null = null;
    if (isRealVideoRef.current) {
      const vidElem = videoRef.current;
      if (!vidElem) return;
      const srcW = vidElem.videoWidth || telemetry?.source_width || 1920;
      const srcH = vidElem.videoHeight || telemetry?.source_height || 1080;
      const scaleX = W / srcW;
      const scaleY = H / srcH;
      const videoFps = telemetry?.video_fps || 29.97;
      const currentVidFrame = Math.floor(vidElem.currentTime * videoFps);

      const frameDets = isLiveStreamingRef.current
        ? (rtFrameBufferRef.current.get(currentVidFrame) || [])
        : (frameMapRef.current.get(currentVidFrame) || []);

      for (const det of frameDets) {
        const bx = Math.min(det.rx, det.rx + det.rw) * scaleX;
        const by = Math.min(det.ry, det.ry + det.rh) * scaleY;
        const bw = Math.max(18, Math.abs(det.rw) * scaleX);
        const bh = Math.max(16, Math.abs(det.rh) * scaleY);
        if (x >= bx - 6 && x <= bx + bw + 6 && y >= by - 6 && y <= by + bh + 6) {
          hitTrk = det.track_id;
          break;
        }
      }
    } else {
      const t = frame / FPS;
      for (const a of V) {
        const s = state(a, t);
        if (s.d > 6 && s.d < 125 && x > s.x - s.bw / 2 - 4 && x < s.x + s.bw / 2 + 4 && y > s.y - s.bh - 4 && y < s.y + 4) {
          hitTrk = a.id;
          break;
        }
      }
    }

    if (hoveredTrackRef.current !== hitTrk) {
      hoveredTrackRef.current = hitTrk;
      setHoveredTrack(hitTrk);
    }
  };

  const handleCanvasMouseLeave = () => {
    if (hoveredTrackRef.current !== null) {
      hoveredTrackRef.current = null;
      setHoveredTrack(null);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const r = cv.getBoundingClientRect();
    const x = ((e.clientX - r.left) * W) / r.width;
    const y = ((e.clientY - r.top) * H) / r.height;

    if (isRealVideoRef.current) {
      const vidElem = videoRef.current;
      if (!vidElem) return;
      const srcW = vidElem.videoWidth || telemetry?.source_width || 1920;
      const srcH = vidElem.videoHeight || telemetry?.source_height || 1080;
      const scaleX = W / srcW;
      const scaleY = H / srcH;
      const videoFps = telemetry?.video_fps || 29.97;
      const currentVidFrame = Math.floor(vidElem.currentTime * videoFps);

      const frameDets = isLiveStreamingRef.current
        ? (rtFrameBufferRef.current.get(currentVidFrame) || [])
        : (frameMapRef.current.get(currentVidFrame) || []);

      let hitTrk: number | null = null;
      frameDets.forEach(det => {
        const bx = Math.min(det.rx, det.rx + det.rw) * scaleX;
        const by = Math.min(det.ry, det.ry + det.rh) * scaleY;
        const bw = Math.max(18, Math.abs(det.rw) * scaleX);
        const bh = Math.max(16, Math.abs(det.rh) * scaleY);
        if (x >= bx - 8 && x <= bx + bw + 8 && y >= by - 8 && y <= by + bh + 8) hitTrk = det.track_id;
      });
      if (hitTrk !== null) {
        if (activeModule === 'examination') {
          setSelectedPersonId(hitTrk);
          setSelTrack(hitTrk);
        } else {
          handleInspectTrack(hitTrk, currentVidFrame);
        }
      } else {
        setSelTrack(null);
        setSelectedPersonId(null);
        setFocusSelected(false);
      }
    } else {
      const t = frame / FPS;
      let hit: number | null = null;
      V.forEach(a => {
        const s = state(a, t);
        if (s.d > 6 && s.d < 125 && x > s.x - s.bw / 2 - 4 && x < s.x + s.bw / 2 + 4 && y > s.y - s.bh - 4 && y < s.y + 4) hit = a.id;
      });
      if (hit) {
        handleInspectTrack(hit, frame);
      } else {
        setSelTrack(null);
        setFocusSelected(false);
      }
    }
  };

  const formatTimestamp = (frameIdx: number, fps: number = 30) => {
    const totalSec = frameIdx / (fps || 30);
    const mins = Math.floor(totalSec / 60);
    const secs = (totalSec % 60).toFixed(1);
    return `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}s`;
  };

  const getLiveDetectionLogs = useCallback(() => {
    const fps = telemetry?.video_fps || 29.97;
    const currentVidFrame = videoRef.current
      ? Math.floor((videoRef.current.currentTime || 0) * fps)
      : (latestInferenceFrame || 0);

    if (isRealVideo) {
      if (isLiveStreamingRef.current) {
        const trackMap = new Map<number, any>();
        rtFrameBufferRef.current.forEach((dets, fIdx) => {
          dets.forEach((d: any) => {
            const existing = trackMap.get(d.track_id);
            if (!existing || fIdx >= existing.lastSeenFrame) {
              const isCalibrated = d.measurementState === 'CALIBRATED_SPEED' ||
                (d.speedKmhCalibrated !== null && d.speedKmhCalibrated !== undefined && d.speedKmhCalibrated > 0);
              trackMap.set(d.track_id, {
                ...d,
                lastSeenFrame: fIdx,
                timestamp: formatTimestamp(fIdx, fps),
                inCurrentFrame: Math.abs(fIdx - currentVidFrame) <= 3,
                isCalibrated
              });
            }
          });
        });

        return Array.from(trackMap.values()).sort((a, b) => b.lastSeenFrame - a.lastSeenFrame);
      }

      if (realTracks && realTracks.length > 0) {
        return realTracks.map((trk: any) => {
          const validSpeeds = (trk.speed_measurements || [])
            .filter((s: any) => (s.error_components?.validity === 'VALID' || !s.error_components?.validity) && (s.smoothed_kmh || s.instantaneous_kmh) > 0)
            .map((s: any) => s.smoothed_kmh ?? s.instantaneous_kmh);

          const currentSm = (trk.speed_measurements || []).find((s: any) => s.frame === currentVidFrame);
          const hasCurrentValidSm = currentSm && (currentSm.smoothed_kmh || currentSm.instantaneous_kmh) > 0 &&
            (currentSm.error_components?.validity === 'VALID' || !currentSm.error_components?.validity);

          let speedVal: number | null = hasCurrentValidSm ? (currentSm.smoothed_kmh ?? currentSm.instantaneous_kmh) : null;
          let measurementState: 'CALIBRATED_SPEED' | 'UNCALIBRATED_MOTION' | 'SPEED_UNAVAILABLE' = 'SPEED_UNAVAILABLE';
          let speedStatus = 'SPEED_UNAVAILABLE';

          if (speedVal !== null) {
            measurementState = 'CALIBRATED_SPEED';
            speedStatus = 'VALID';
          } else if (realCalib) {
            if (validSpeeds.length > 0) {
              speedVal = validSpeeds[validSpeeds.length - 1];
              measurementState = 'CALIBRATED_SPEED';
              speedStatus = 'VALID';
            } else {
              speedStatus = currentSm?.error_components?.validity || 'OUT_OF_ROI';
              measurementState = 'SPEED_UNAVAILABLE';
            }
          } else if (trk.trajectory && trk.trajectory.length >= 2) {
            measurementState = 'UNCALIBRATED_MOTION';
            speedStatus = 'UNCALIBRATED_MOTION';
          } else {
            speedStatus = 'CALIBRATION_REQUIRED';
            measurementState = 'SPEED_UNAVAILABLE';
          }

          const inCurrentFrame = currentVidFrame >= (trk.first_frame || 0) && currentVidFrame <= (trk.last_frame || 999999);

          return {
            track_id: trk.track_id,
            vehicle_class: trk.vehicle_class || 'Vehicle',
            confidence: trk.confidence || 0.95,
            speedKmh: speedVal,
            speedKmhCalibrated: measurementState === 'CALIBRATED_SPEED' ? speedVal : null,
            measurementState,
            speedStatus,
            isCalibrated: measurementState === 'CALIBRATED_SPEED',
            timestamp: formatTimestamp(trk.first_frame || 0, fps),
            lastSeenFrame: trk.last_frame || 0,
            inCurrentFrame
          };
        });
      }

      return [];
    }

    return V.map(a => {
      const spd = estSpeed(a, frame);
      return {
        track_id: a.id,
        vehicle_class: a.cls || 'Vehicle',
        confidence: a.conf || 0.92,
        speedKmh: spd,
        speedKmhCalibrated: spd,
        measurementState: 'CALIBRATED_SPEED',
        speedStatus: 'VALID',
        isCalibrated: true,
        timestamp: formatTimestamp(frame, FPS),
        lastSeenFrame: frame,
        inCurrentFrame: true
      };
    });
  }, [telemetry, isRealVideo, latestInferenceFrame, realTracks, realCalib, frame]);

  const renderLiveDetectionLog = () => {
    const logs = getLiveDetectionLogs();
    const filteredLogs = logs.filter(l => {
      if (logFilter === 'speeding') return l.isCalibrated && l.speedKmh !== null && l.speedKmh > speedLimit;
      if (logFilter === 'active') return l.inCurrentFrame;
      return true;
    });

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Filters and Counters */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
          <div style={{ display: 'flex', gap: 4 }}>
            {(['all', 'active', 'speeding'] as const).map(f => (
              <button
                key={f}
                className={`tab-pill ${logFilter === f ? 'active' : 'inactive'}`}
                style={{ fontSize: '10px', padding: '4px 8px' }}
                onClick={() => setLogFilter(f)}
              >
                {f === 'all' ? `ALL (${logs.length})` : f === 'active' ? `LIVE (${logs.filter(x => x.inCurrentFrame).length})` : `SPEEDING (${logs.filter(x => x.isCalibrated && x.speedKmh && x.speedKmh > speedLimit).length})`}
              </button>
            ))}
          </div>
          {isLiveStreamingRef.current && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#10B981', fontWeight: 700 }}>
              <span className="pulse-dot" /> STREAMING
            </div>
          )}
        </div>

        {/* Live Detection Log Table */}
        <div className="log-table-container">
          <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse', color: '#E2E8F0' }}>
            <thead>
              <tr style={{ background: '#111827', color: '#94A3B8', textAlign: 'left', borderBottom: '1px solid #1E293B', position: 'sticky', top: 0, zIndex: 1 }}>
                <th style={{ padding: '8px 6px' }}>VEHICLE</th>
                <th style={{ padding: '8px 6px' }}>CLASS</th>
                <th style={{ padding: '8px 6px' }}>SPEED</th>
                <th style={{ padding: '8px 6px' }}>CONF</th>
                <th style={{ padding: '8px 6px' }}>TIME</th>
              </tr>
            </thead>
            <tbody>
              {filteredLogs.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '28px 8px', color: '#64748B' }}>
                    {isLiveStreamingRef.current ? 'Awaiting real-time vehicle detections...' : 'No detections match current filter.'}
                  </td>
                </tr>
              ) : (
                filteredLogs.map(log => {
                  const isSelected = selTrack === log.track_id;
                  const isCalibrated = log.isCalibrated || log.measurementState === 'CALIBRATED_SPEED';
                  const isSpeeding = isCalibrated && log.speedKmh !== null && log.speedKmh > speedLimit;
                  const isUncalibrated = log.measurementState === 'UNCALIBRATED_MOTION' ||
                    log.speedStatus === 'UNCALIBRATED_MOTION' || log.speedStatus === 'UNCALIBRATED_ESTIMATE';

                  let speedDisplay = 'Speed unavailable';
                  let speedColor = '#94A3B8';

                  if (isCalibrated && log.speedKmh !== null) {
                    speedDisplay = `${Math.round(log.speedKmh)} km/h`;
                    speedColor = isSpeeding ? '#EF4444' : '#10B981';
                  } else if (isUncalibrated) {
                    speedDisplay = 'Uncalibrated motion';
                    speedColor = '#0EA5E9';
                  } else if (log.speedStatus === 'CALIBRATION_REQUIRED') {
                    speedDisplay = 'No Calib';
                    speedColor = '#64748B';
                  } else {
                    speedDisplay = 'Calc...';
                    speedColor = '#F59E0B';
                  }

                  return (
                    <tr
                      key={log.track_id}
                      onClick={() => {
                        handleInspectTrack(log.track_id);
                      }}
                      style={{
                        background: isSelected ? 'rgba(245, 158, 11, 0.2)' : log.inCurrentFrame ? 'rgba(16, 185, 129, 0.08)' : 'transparent',
                        borderBottom: '1px solid #1E293B',
                        cursor: 'pointer',
                        transition: 'background 0.15s'
                      }}
                      title="Click to inspect this vehicle track"
                    >
                      <td style={{ padding: '8px 6px', fontWeight: 700, color: '#F8FAFC' }}>
                        {log.inCurrentFrame && <span className="pulse-dot" style={{ width: 6, height: 6, marginRight: 6 }} />}
                        #{log.track_id}
                      </td>
                      <td style={{ padding: '8px 6px' }}>
                        <span style={{
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '10px',
                          fontWeight: 700,
                          background: log.vehicle_class === 'Truck' ? '#334155' : log.vehicle_class === 'Motorcycle' ? '#4C1D95' : '#1E293B',
                          color: '#E0E7FF'
                        }}>
                          {log.vehicle_class}
                        </span>
                      </td>
                      <td style={{ padding: '8px 6px', fontWeight: 800, color: speedColor }}>
                        {speedDisplay}
                      </td>
                      <td style={{ padding: '8px 6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ fontSize: '10px', color: '#94A3B8' }}>{((log.confidence || 0.9) * 100).toFixed(0)}%</span>
                          <div style={{ width: 24, height: 4, background: '#1E293B', borderRadius: 2, overflow: 'hidden' }}>
                            <div style={{ width: `${Math.round((log.confidence || 0.9) * 100)}%`, height: '100%', background: '#4F46E5' }} />
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: '8px 6px', color: '#64748B', fontFamily: 'monospace' }}>
                        {log.timestamp}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: '11px', color: '#64748B', textAlign: 'center' }}>
          💡 Click any row to view track telemetry & speed curve
        </div>
      </div>
    );
  };

  // =========================================================================
  // Inspector Panel — Wording per Requirement #13
  // =========================================================================
  const renderInspector = () => {
    if (isRealVideo) {
      // Check final completed tracks first, then live frame buffer
      const finalTrk = selTrack !== null
        ? realTracks.find((t: any) => t.track_id === selTrack)
        : (realTracks.length > 0 ? realTracks[0] : null);

      // Collect all live detections for the selected or first tracked vehicle
      const currentTrackId = selTrack ?? finalTrk?.track_id;
      const allLiveInstances = Array.from(rtFrameBufferRef.current.entries())
        .map(([fIdx, dets]) => ({
          frame: fIdx,
          det: dets.find((d: any) => d.track_id === (currentTrackId ?? dets[0]?.track_id))
        }))
        .filter(item => item.det)
        .sort((a, b) => a.frame - b.frame);

      const latestLive = allLiveInstances.length > 0 ? allLiveInstances[allLiveInstances.length - 1].det : null;
      const targetTrackId = finalTrk?.track_id ?? latestLive?.track_id;

      if (!finalTrk && !latestLive) {
        return (
          <div style={{ padding: '12px', background: '#F8FAFC', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
            <div style={{ color: isLiveStreamingRef.current ? '#10B981' : '#4F46E5', fontWeight: 700, fontSize: '13px' }}>
              {isLiveStreamingRef.current ? '● LIVE DETECTION ACTIVE' : 'LIVE DETECTION'}
            </div>
            <div style={{ margin: '8px 0', fontSize: '12px', color: '#64748B' }}>
              {liveTrackCount > 0 ? `Tracking ${liveTrackCount} vehicles in real-time.` : 'Waiting for vehicle detections...'}
            </div>
            <div style={{ fontSize: '11px', color: '#94A3B8' }}>Click any vehicle bounding box on the video overlay to inspect track details.</div>
          </div>
        );
      }

      // Determine speed and history for the active track
      let speedKmh = 0;
      let uncertaintyKmh = 0;
      let statusStr = 'CALCULATING...';
      let vehicleClass = finalTrk?.vehicle_class || latestLive?.vehicle_class || 'Car';
      let confidence = finalTrk?.confidence || latestLive?.confidence || 0.95;
      let hist: number[] = [];

      if (finalTrk) {
        const currentVidFrame = videoRef.current ? Math.floor(videoRef.current.currentTime * (telemetry?.video_fps || 29.97)) : 0;
        let sm = (finalTrk.speed_measurements || []).find((s: any) => s.frame === currentVidFrame);
        if (!sm) {
          const validSms = (finalTrk.speed_measurements || []).filter((s: any) => s.smoothed_kmh !== null && s.smoothed_kmh > 0);
          sm = validSms.length > 0 ? validSms[validSms.length - 1] : finalTrk.speed_measurements?.[finalTrk.speed_measurements.length - 1];
        }

        speedKmh = sm?.smoothed_kmh ?? sm?.instantaneous_kmh ?? 0;
        uncertaintyKmh = sm?.uncertainty_kmh || 0;
        const val = sm?.error_components?.validity;
        if (speedKmh > 0) {
          statusStr = 'VALID';
        } else if (val === 'CALIBRATION_REQUIRED') {
          statusStr = 'CALIBRATION REQUIRED';
        } else if (val === 'OUT_OF_ROI') {
          statusStr = 'OUT OF ROI';
        } else if (val === 'INSUFFICIENT_HISTORY' || val === 'INSUFFICIENT_DATA') {
          statusStr = 'INSUFFICIENT TRACK HISTORY';
        } else {
          statusStr = val || 'CALCULATING...';
        }

        hist = (finalTrk.speed_measurements || [])
          .map((s: any) => s.smoothed_kmh || s.instantaneous_kmh)
          .filter((v: any) => v !== null && v !== undefined && v > 0);
      } else if (latestLive) {
        speedKmh = latestLive.speedKmh || 0;
        uncertaintyKmh = latestLive.speedUncertainty || 0;
        statusStr = latestLive.speedStatus || 'CALCULATING...';
        if (statusStr === 'CALIBRATION_REQUIRED') statusStr = 'CALIBRATION REQUIRED';
        if (statusStr === 'INSUFFICIENT_DATA' || statusStr === 'INSUFFICIENT_HISTORY') statusStr = 'CALCULATING...';
        hist = allLiveInstances
          .map(item => item.det.speedKmh)
          .filter((spd: any) => spd !== null && spd !== undefined && spd > 0);
      }

      const isBad = speedKmh > speedLimit;
      const isCalibReq = statusStr === 'CALIBRATION REQUIRED' || statusStr === 'CALIBRATION_REQUIRED';

      return (
        <div>
          <div className="row"><b>Track #{targetTrackId}</b><span className="badge">{vehicleClass}</span></div>
          <div className="row mu"><span>Confidence</span><span>{(confidence * 100).toFixed(1)}%</span></div>
          <div className="big" style={{ color: speedKmh > 0 ? (isBad ? '#EF4444' : '#10B981') : '#F59E0B' }}>
            {speedKmh > 0 ? speedKmh.toFixed(1) : statusStr} {speedKmh > 0 && <small>km/h ± {uncertaintyKmh.toFixed(1)}</small>}
          </div>
          <div className="mu">
            {speedKmh > 0 ? 'Kalman-Smoothed Monocular Speed Estimation' : isCalibReq ? 'Camera calibration required for metric speed' : 'Accumulating space-time observations'}
          </div>
          <canvas id="sp" width={300} height={110} style={{ margin: '8px 0' }} ref={node => {
            if (!node) return; const c = node.getContext('2d'); if (!c) return;
            c.fillStyle = '#F8FAFC'; c.fillRect(0, 0, 300, 110);
            if (hist.length >= 2) {
              const lo = Math.min(...hist, speedLimit) - 5;
              const hi = Math.max(...hist, speedLimit) + 5;
              const Y = (q: number) => 105 - ((q - lo) / (hi - lo || 1)) * 100;
              c.strokeStyle = '#EF4444'; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(0, Y(speedLimit)); c.lineTo(300, Y(speedLimit)); c.stroke(); c.setLineDash([]);
              c.strokeStyle = '#4F46E5'; c.lineWidth = 2; c.beginPath();
              hist.forEach((q: number, i: number) => i ? c.lineTo((i * 300) / (hist.length - 1), Y(q)) : c.moveTo(0, Y(q)));
              c.stroke();
              c.fillStyle = '#64748B'; c.font = '10px sans-serif'; c.fillText('km/h vs timeline', 4, 12);
              c.fillStyle = '#10B981'; c.fillText(`${speedKmh.toFixed(1)} km/h`, 240, 12);
            } else {
              c.fillStyle = '#94A3B8'; c.font = '11px sans-serif'; c.textAlign = 'center';
              const notice = isCalibReq ? 'CALIBRATION REQUIRED FOR METRIC SPEED' : 'ACCUMULATING SPEED SAMPLES...';
              c.fillText(notice, 150, 58);
            }
          }} />
        </div>
      );
    }

    const a = V.find(v => v.id === selTrack) || V[0];
    const spd = estSpeed(a, frame);
    return (
      <div>
        <div className="row"><b>Track #{a.id}</b><span className="badge">{a.cls}</span></div>
        <div className="row mu"><span>Confidence</span><span>{(a.conf * 100).toFixed(1)}%</span></div>
        <div className="big" style={{ color: spd > speedLimit ? '#EF4444' : '#10B981' }}>
          {spd.toFixed(1)} <small>km/h ± {(spd * 0.05).toFixed(1)}</small>
        </div>
        <div className="mu" style={{ color: '#F59E0B', fontWeight: 700, marginTop: 4 }}>DEMO MODE — SYNTHETIC · Not real detection</div>
      </div>
    );
  };

  // =========================================================================
  // Pipeline Status Component (Requirement #8)
  // =========================================================================
  const renderPipelineStatus = () => {
    const stages: { key: PipelineStage, label: string }[] = [
      { key: 'VIDEO_READY', label: 'Video Ready' },
      { key: 'UPLOADING', label: 'Uploading' },
      { key: 'JOB_CREATED', label: 'Job Created' },
      { key: 'PROCESSING', label: 'Processing' },
      { key: 'LIVE_INFERENCE', label: '● Live Inference' },
      { key: 'FINALIZING', label: 'Finalizing' },
      { key: 'COMPLETED', label: 'Completed' },
    ];
    const stageOrder: PipelineStage[] = [
      'IDLE', 'VIDEO_READY', 'UPLOADING', 'UPLOAD_COMPLETE', 'JOB_CREATED',
      'PROCESSING', 'LIVE_INFERENCE', 'FINALIZING', 'COMPLETED'
    ];
    const currentIdx = stageOrder.indexOf(pipelineStage);

    if (pipelineStage === 'IDLE' || pipelineStage === 'COMPLETED') return null;

    return (
      <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap', margin: '8px 0', padding: '8px 12px', background: '#0F172A', borderRadius: 8 }}>
        {stages.map(({ key, label }, idx) => {
          const sIdx = stageOrder.indexOf(key);
          const isDone = sIdx < currentIdx;
          const isActive = sIdx === currentIdx;
          return (
            <React.Fragment key={key}>
              <span style={{
                fontSize: 11, fontWeight: isActive ? 800 : 500, padding: '2px 8px', borderRadius: 12,
                background: isActive ? '#4F46E5' : isDone ? '#10B981' : '#1E293B',
                color: isActive ? '#fff' : isDone ? '#fff' : '#475569',
                transition: 'all 0.3s'
              }}>{label}</span>
              {idx < stages.length - 1 && <span style={{ color: '#334155', fontSize: 10 }}>›</span>}
            </React.Fragment>
          );
        })}
        {pipelineStage === 'LIVE_INFERENCE' && (
          <span style={{ marginLeft: 'auto', fontSize: 11, color: '#10B981', fontWeight: 700 }}>
            {liveTrackCount} vehicles tracked
          </span>
        )}
      </div>
    );
  };

  // =========================================================================
  // Phase 22: Synchronized Vehicle Intelligence Table Data
  // =========================================================================
  const getVehicleTableData = () => {
    if (isRealVideo) {
      if (realTracks && realTracks.length > 0) {
        const currentVidFrame = videoRef.current
          ? Math.floor((videoRef.current.currentTime || 0) * (telemetry?.video_fps || 29.97))
          : 0;

        return realTracks.map((trk: any) => {
          const trajLen = trk.trajectory?.length || trk.detection_count || 1;
          const isSustained = trajLen >= 5;

          const validSpeeds = (trk.speed_measurements || [])
            .filter((s: any) => (s.error_components?.validity === 'VALID' || !s.error_components?.validity) && (s.smoothed_kmh || s.instantaneous_kmh) > 0)
            .map((s: any) => s.smoothed_kmh ?? s.instantaneous_kmh);

          const currentSm = (trk.speed_measurements || []).find((s: any) => s.frame === currentVidFrame);
          const hasCurrentValidSm = currentSm && (currentSm.smoothed_kmh || currentSm.instantaneous_kmh) > 0 &&
            (currentSm.error_components?.validity === 'VALID' || !currentSm.error_components?.validity);

          let currentSpeedVal: number | null = hasCurrentValidSm ? (currentSm.smoothed_kmh ?? currentSm.instantaneous_kmh) : null;
          let measurementState: 'CALIBRATED_SPEED' | 'UNCALIBRATED_MOTION' | 'SPEED_UNAVAILABLE' = 'SPEED_UNAVAILABLE';
          let unavailReason = '';

          if (currentSpeedVal !== null) {
            measurementState = 'CALIBRATED_SPEED';
          } else if (realCalib) {
            if (validSpeeds.length > 0) {
              currentSpeedVal = validSpeeds[validSpeeds.length - 1];
              measurementState = 'CALIBRATED_SPEED';
            } else {
              measurementState = 'SPEED_UNAVAILABLE';
              if (trajLen < 3) {
                unavailReason = 'Duration < 3 frames';
              } else {
                const hasOutOfRoi = (trk.speed_measurements || []).some((s: any) => s.error_components?.validity === 'OUT_OF_ROI');
                unavailReason = hasOutOfRoi ? 'Outside road trapezoid' : 'Insufficient observations';
              }
            }
          } else if (trk.trajectory && trk.trajectory.length >= 2) {
            measurementState = 'UNCALIBRATED_MOTION';
            unavailReason = 'Road calibration required';
          } else {
            measurementState = 'SPEED_UNAVAILABLE';
            unavailReason = 'Road calibration required';
          }

          const isCalibrated = measurementState === 'CALIBRATED_SPEED';
          const avgSpeed = isCalibrated && validSpeeds.length > 0
            ? (validSpeeds.reduce((a: number, b: number) => a + b, 0) / validSpeeds.length).toFixed(1) + ' km/h'
            : '-';
          const maxSpeed = isCalibrated && validSpeeds.length > 0
            ? Math.max(...validSpeeds).toFixed(1) + ' km/h'
            : '-';

          let calibStatus = realCalib ? 'CALIBRATED' : 'UNCALIBRATED';
          let validity = isCalibrated ? 'VALID' : measurementState === 'UNCALIBRATED_MOTION' ? 'UNCALIBRATED' : (unavailReason || 'UNAVAILABLE');
          let currentSpeedText = isCalibrated && currentSpeedVal !== null
            ? `${Math.round(currentSpeedVal)} km/h`
            : `- (${unavailReason || 'Speed unavailable'})`;

          const isCurrentlyInFrame = currentVidFrame >= (trk.first_frame || 0) && currentVidFrame <= (trk.last_frame || 999999);
          const trackStatus = isCurrentlyInFrame ? 'IN FRAME' : isSustained ? 'SUSTAINED' : 'LOGGED';

          return {
            track_id: trk.track_id,
            vehicle_class: trk.vehicle_class || 'Vehicle',
            current_speed: currentSpeedText,
            current_speed_raw: isCalibrated ? currentSpeedVal : null,
            isCalibrated,
            isSustained,
            trajectory_length: trajLen,
            measurement_state: measurementState,
            avg_speed: avgSpeed,
            max_speed: maxSpeed,
            track_status: trackStatus,
            calibration_status: calibStatus,
            validity: validity
          };
        });
      }

      if (rtFrameBufferRef.current.size > 0) {
        const trackMap = new Map<number, any>();
        rtFrameBufferRef.current.forEach((dets) => {
          dets.forEach((d: any) => {
            trackMap.set(d.track_id, d);
          });
        });

        return Array.from(trackMap.values()).map((d: any) => {
          const isCalibrated = d.measurementState === 'CALIBRATED_SPEED' ||
            (d.speedKmhCalibrated !== null && d.speedKmhCalibrated !== undefined && d.speedKmhCalibrated > 0);
          const isUncalibrated = d.measurementState === 'UNCALIBRATED_MOTION';
          const speedDisplay = isCalibrated && d.speedKmh !== null
            ? `${Math.round(d.speedKmh)} km/h`
            : isUncalibrated
            ? 'Uncalibrated motion'
            : 'Speed unavailable';

          return {
            track_id: d.track_id,
            vehicle_class: d.vehicle_class || 'Vehicle',
            current_speed: speedDisplay,
            current_speed_raw: isCalibrated ? d.speedKmh : null,
            isCalibrated,
            measurement_state: d.measurementState || (isCalibrated ? 'CALIBRATED_SPEED' : isUncalibrated ? 'UNCALIBRATED_MOTION' : 'SPEED_UNAVAILABLE'),
            avg_speed: isCalibrated && d.avgSpeedKmh ? `${d.avgSpeedKmh.toFixed(1)} km/h` : '-',
            max_speed: isCalibrated && d.maxSpeedKmh ? `${d.maxSpeedKmh.toFixed(1)} km/h` : '-',
            track_status: 'ACTIVE',
            calibration_status: realCalib ? 'CALIBRATED' : 'UNCALIBRATED',
            validity: isCalibrated ? 'CALIBRATED_SPEED' : isUncalibrated ? 'UNCALIBRATED_MOTION' : 'SPEED_UNAVAILABLE'
          };
        });
      }

      return [];
    }

    return V.map(a => {
      const spd = estSpeed(a, frame);
      return {
        track_id: a.id,
        vehicle_class: a.cls || 'Vehicle',
        current_speed: `${Math.round(spd)} km/h`,
        current_speed_raw: spd,
        isCalibrated: true,
        measurement_state: 'CALIBRATED_SPEED',
        avg_speed: `${Math.round(a.v0 || 60)} km/h`,
        max_speed: `${Math.round((a.v0 || 60) * 1.15)} km/h`,
        track_status: 'SIMULATED',
        calibration_status: 'SYNTHETIC',
        validity: 'CALIBRATED_SPEED'
      };
    });
  };

  // =========================================================================
  // JSX Layout
  // =========================================================================
  return (
    <div className="grid">
      <div>
        <div className="card">
          <h3 style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
            <span>Analytical Workbench</span>{' '}
            <span className="badge">{srcName}</span>
            {activeJobId && (
              <span className="badge" style={{ background: '#0F172A', color: '#10B981', fontWeight: 700 }}>
                ACTIVE JOB: #{activeJobId}
              </span>
            )}
            {isLiveStreamingRef.current && (
              <span className="badge" style={{ background: '#D1FAE5', color: '#065F46' }}>● REAL VIDEO — LIVE AI</span>
            )}
            {pipelineStage === 'COMPLETED' && (
              <span className="badge" style={{ background: '#EEF2FF', color: '#4F46E5' }}>REAL VIDEO · LIVE AI</span>
            )}
            {!isRealVideo && (
              <span className="badge" style={{ background: '#FFFBEB', color: '#B45309' }}>DEMO MODE — SYNTHETIC</span>
            )}

            <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', color: '#94A3B8', fontWeight: 700 }}>DOMAIN:</span>
              <button
                className={`tab-pill ${activeModule === 'traffic' ? 'active' : 'inactive'}`}
                style={{
                  background: activeModule === 'traffic' ? '#2563EB' : '#1E293B',
                  color: activeModule === 'traffic' ? '#FFFFFF' : '#94A3B8',
                  padding: '3px 10px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: 700,
                  cursor: 'pointer',
                  border: 'none'
                }}
                onClick={() => setActiveModule('traffic')}
              >
                🚗 Traffic
              </button>
              <button
                className={`tab-pill ${activeModule === 'examination' ? 'active' : 'inactive'}`}
                style={{
                  background: activeModule === 'examination' ? '#4F46E5' : '#1E293B',
                  color: activeModule === 'examination' ? '#FFFFFF' : '#94A3B8',
                  padding: '3px 10px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: 700,
                  cursor: 'pointer',
                  border: 'none'
                }}
                onClick={() => setActiveModule('examination')}
              >
                🎓 Examination
              </button>
              <button
                style={{
                  background: '#10B981',
                  color: '#FFFFFF',
                  padding: '3px 10px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: 700,
                  cursor: 'pointer',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
                onClick={() => openExportModal(activeJobId || sharedJobId || undefined)}
                title="Export this job to Excel (.xlsx)"
              >
                <span>📊</span>
                <span>Export to Excel</span>
              </button>
              {/* Gemini AI Status Badge */}
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  fontWeight: 600,
                  padding: '3px 8px',
                  borderRadius: '4px',
                  background: geminiStatus?.status === 'READY' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(100, 116, 139, 0.15)',
                  color: geminiStatus?.status === 'READY' ? '#10B981' : '#94A3B8',
                  border: `1px solid ${geminiStatus?.status === 'READY' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(100, 116, 139, 0.3)'}`
                }}
                title={geminiStatus?.detail || 'Gemini Multimodal Intelligence'}
              >
                <span>✨</span>
                <span>Gemini {geminiStatus?.status === 'READY' ? 'ONLINE' : geminiStatus?.status || 'OFFLINE'}</span>
              </span>
            </div>
          </h3>

          {/* Scene Router Recommendation Banner */}
          {sceneRec && (
            <SceneRecommendationBanner
              recommendation={sceneRec}
              onAccept={startJobForModule}
              onSelectModule={startJobForModule}
              onDismiss={() => setSceneRec(null)}
            />
          )}


          <video
            ref={videoRef}
            src={videoSrc || undefined}
            playsInline
            muted
            preload="auto"
            onPlay={() => setPlaying(true)}
            onPause={() => { if (!isLiveStreamingRef.current) setPlaying(false); }}
            onEnded={() => {
              if (videoRef.current?.loop) {
                videoRef.current.currentTime = 0;
                videoRef.current.play().catch(() => {});
              }
            }}
            onLoadedMetadata={() => {
              if (videoRef.current && videoRef.current.duration) {
                const fps = telemetry?.video_fps || 29.97;
                setTelemetry((prev: any) => ({
                  ...(prev || {}),
                  total_frames: prev?.total_frames || Math.floor(videoRef.current!.duration * fps),
                  video_duration_sec: prev?.video_duration_sec || videoRef.current!.duration
                }));
              }
              videoRef.current?.play().catch(() => {});
            }}
            onSeeked={() => {
              setRenderTick(t => t + 1);
              if (canvasRef.current && videoRef.current) {
                // Re-trigger render on seek
                setFrame(Math.floor((videoRef.current.currentTime || 0) * (telemetry?.video_fps || 29.97)));
              }
            }}
            style={{
              position: 'absolute',
              width: 320,
              height: 180,
              opacity: 0.01,
              pointerEvents: 'none',
              zIndex: -1
            }}
          />
          {/* Crowded-Scene Intelligent Visualization Toolbar (Requirement #2) */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '8px',
            marginBottom: '8px',
            padding: '6px 12px',
            background: '#0F172A',
            borderRadius: '6px',
            border: '1px solid #1E293B',
            color: '#F8FAFC',
            fontSize: '11px'
          }}>
            {/* Mode Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#94A3B8', fontWeight: 700 }}>VIZ MODE:</span>
              {(['MINIMAL', 'STANDARD', 'INTELLIGENCE'] as const).map(m => (
                <button
                  key={m}
                  onClick={() => setVizMode(m)}
                  style={{
                    background: vizMode === m ? '#4F46E5' : '#1E293B',
                    color: vizMode === m ? '#FFFFFF' : '#94A3B8',
                    border: `1px solid ${vizMode === m ? '#6366F1' : '#334155'}`,
                    borderRadius: '4px',
                    padding: '3px 8px',
                    fontSize: '10px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                  title={m === 'MINIMAL' ? 'Boxes only (labels on hover/select)' : m === 'STANDARD' ? 'Compact class and ID (speed only if calibrated)' : 'Full measurements, confidence, and trajectories'}
                >
                  {m}
                </button>
              ))}
            </div>

            {/* Quick Toggles */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={e => setShowLabels(e.target.checked)}
                />
                <span style={{ color: showLabels ? '#E2E8F0' : '#64748B' }}>Labels</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showTrajectories}
                  onChange={e => setShowTrajectories(e.target.checked)}
                />
                <span style={{ color: showTrajectories ? '#E2E8F0' : '#64748B' }}>Trails</span>
              </label>

              {selTrack !== null && (
                <button
                  onClick={() => setFocusSelected(!focusSelected)}
                  style={{
                    background: focusSelected ? '#F59E0B' : '#1E293B',
                    color: focusSelected ? '#000000' : '#F59E0B',
                    border: '1px solid #F59E0B',
                    borderRadius: '4px',
                    padding: '2px 7px',
                    fontSize: '10px',
                    fontWeight: 700,
                    cursor: 'pointer'
                  }}
                  title="Isolate selected track"
                >
                  {focusSelected ? `Focus: #${selTrack} (Active)` : `Focus: #${selTrack}`}
                </button>
              )}
            </div>

            {/* Filters: Class, Confidence, Density */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ color: '#94A3B8' }}>Class:</span>
                <select
                  value={classFilter}
                  onChange={e => setClassFilter(e.target.value as any)}
                  style={{
                    background: '#1E293B',
                    color: '#F8FAFC',
                    border: '1px solid #334155',
                    borderRadius: '4px',
                    padding: '2px 4px',
                    fontSize: '10px'
                  }}
                >
                  <option value="ALL">All Classes</option>
                  <option value="CAR">Cars</option>
                  <option value="TRUCK">Trucks</option>
                  <option value="BUS">Buses</option>
                  <option value="MOTORCYCLE">Motorcycles</option>
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ color: '#94A3B8' }}>Min Conf:</span>
                <select
                  value={minConfidence}
                  onChange={e => setMinConfidence(Number(e.target.value))}
                  style={{
                    background: '#1E293B',
                    color: '#F8FAFC',
                    border: '1px solid #334155',
                    borderRadius: '4px',
                    padding: '2px 4px',
                    fontSize: '10px'
                  }}
                >
                  <option value={0.15}>&gt; 15%</option>
                  <option value={0.25}>&gt; 25%</option>
                  <option value={0.50}>&gt; 50%</option>
                  <option value={0.75}>&gt; 75%</option>
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ color: '#94A3B8' }}>Max Badges:</span>
                <select
                  value={maxLabels}
                  onChange={e => setMaxLabels(Number(e.target.value))}
                  style={{
                    background: '#1E293B',
                    color: '#F8FAFC',
                    border: '1px solid #334155',
                    borderRadius: '4px',
                    padding: '2px 4px',
                    fontSize: '10px'
                  }}
                >
                  <option value={8}>8 max</option>
                  <option value={14}>14 max</option>
                  <option value={24}>24 max</option>
                  <option value={999}>All</option>
                </select>
              </div>
            </div>
          </div>

          <canvas
            ref={canvasRef}
            width={W}
            height={H}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
            style={{ cursor: 'pointer', display: 'block', borderRadius: 6 }}
          />

          {/* Pipeline Progress Status */}
          {renderPipelineStatus()}

          {/* Status Bar */}
          {jobStatusText && (
            <div style={{
              background: pipelineStage === 'LIVE_INFERENCE' ? '#0F172A' : '#EEF2FF',
              border: `1px solid ${pipelineStage === 'LIVE_INFERENCE' ? '#10B981' : '#C7D2FE'}`,
              color: pipelineStage === 'LIVE_INFERENCE' ? '#10B981' : '#4F46E5',
              padding: '6px 10px', borderRadius: '6px', fontSize: '12px', fontWeight: 600, marginTop: '8px'
            }}>
              {jobStatusText}
            </div>
          )}

          {/* Latency & Frame Sync Debug Readout (Requirement #12) */}
          <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#64748B', marginTop: 6, flexWrap: 'wrap' }}>
            {videoRef.current && (
              <span>Video Frame: <b style={{ color: '#3B82F6' }}>{Math.floor((videoRef.current.currentTime || 0) * (telemetry?.video_fps || 29.97))}</b></span>
            )}
            {latestInferenceFrame !== null && (
              <span>Inference Frame: <b style={{ color: '#10B981' }}>{latestInferenceFrame}</b></span>
            )}
            {serverPipelineLatencyMs !== null && (
              <span>Server Pipeline Latency: <b style={{ color: serverPipelineLatencyMs <= 100 ? '#10B981' : '#F59E0B' }}>{serverPipelineLatencyMs} ms</b></span>
            )}
            {firstBoxMs && <span>🎯 First box: <b style={{ color: '#10B981' }}>{firstBoxMs} ms</b></span>}
            {firstDetectionMs && <span>⚡ First detection: <b style={{ color: '#4F46E5' }}>{firstDetectionMs} ms</b></span>}
            {processingFps && <span>🚀 FPS: <b style={{ color: '#0EA5E9' }}>{processingFps.toFixed(1)}</b></span>}
            {realTimeFactor !== null && <span>⏱ RTF: <b style={{ color: realTimeFactor >= 1.0 ? '#10B981' : '#F59E0B' }}>{realTimeFactor.toFixed(2)}x</b></span>}
            <span>Calibration: <b style={{ color: realCalib ? '#10B981' : '#F59E0B' }}>{realCalib ? `ACTIVE (#${realCalib.id})` : 'NOT CONFIGURED'}</b></span>
          </div>

          {/* Controls */}
          <div className="bar">
            <button className="b" onClick={handleTogglePlay}>{playing ? 'Pause' : 'Play'}</button>
            <button className="b g" onClick={() => handleStepFrame(-1)}>◀ −1</button>
            <button className="b g" onClick={() => handleStepFrame(1)}>+1 ▶</button>
            <input
              type="range"
              min="0"
              max={Math.max(1, totalFrames - 1)}
              value={Math.min(frame, Math.max(1, totalFrames - 1))}
              onChange={e => handleSeek(Number(e.target.value))}
            />
            <span className="mu">
              {Math.floor((frame / (isRealVideo ? (telemetry?.video_fps || 29.97) : FPS)) / 60)}:
              {((frame / (isRealVideo ? (telemetry?.video_fps || 29.97) : FPS)) % 60).toFixed(2).padStart(5, '0')} · f{frame} / {totalFrames}
            </span>
          </div>

          <div className="bar">
            {activeModule === 'traffic' ? (
              <label>Limit{' '}<input type="number" value={speedLimit} onChange={e => { setSpeedLimit(Number(e.target.value)); seenRef.current.clear(); }} />{' '}km/h</label>
            ) : (
              <span style={{ color: '#818CF8', fontWeight: 600, fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                🎓 Examination Intelligence Active · Observable Events Proctorship
              </span>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 700 }}>Profile:</span>
              <select
                value={processingMode}
                onChange={e => setProcessingMode(e.target.value as ProcessingMode)}
                style={{
                  padding: '3px 8px',
                  fontSize: '11px',
                  borderRadius: '4px',
                  border: '1px solid #CBD5E1',
                  background: processingMode === 'high_speed' ? '#EEF2FF' : '#F8FAFC',
                  color: processingMode === 'high_speed' ? '#4F46E5' : '#334155',
                  fontWeight: 600
                }}
                title="Select inference speed profile (Optimized for Intel i5 CPU)"
              >
                <option value="high_speed">⚡ High Speed (20+ FPS, Stride 3)</option>
                <option value="balanced">⚖️ Balanced (~15 FPS, Stride 2)</option>
                <option value="forensic">🔬 Deep Forensic (Dense 1:1)</option>
              </select>
            </div>
            <button
              className="b g"
              onClick={handleSaveWorkbenchCalibration}
              style={{
                background: realCalib ? '#065F46' : '#2563EB',
                color: '#FFF',
                padding: '4px 10px',
                fontSize: '11px',
                fontWeight: 700,
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
              title="Apply road geometry homography calibration for metric speed estimation"
            >
              📐 {realCalib ? `Road Calibrated (#${realCalib.id || 'Active'})` : 'Calibrate Road (Homography)'}
            </button>
            <label>Upload MP4/WebM{' '}<input type="file" accept="video/mp4,video/webm" onChange={handleFileUpload} /></label>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#64748B', marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #E2E8F0', flexWrap: 'wrap' }}>
            <span>Detector: <b style={{ color: '#4F46E5' }}>{activeModule === 'traffic' ? (telemetry?.detector_actual || 'YOLOX-Nano-ONNX') : 'YOLOX-COCO (Person/Objects)'}</b></span>
            <span>Tracker: <b style={{ color: '#10B981' }}>{activeModule === 'traffic' ? 'ByteTrack' : 'PersonTracker (Spatial IoU)'}</b></span>
            <span>Profile: <b style={{ color: '#0EA5E9' }}>{processingMode.toUpperCase()}</b></span>
            <span>Status: <b style={{ color: wsConnected ? '#10B981' : '#64748B' }}>{wsConnected ? '● STREAMING' : pipelineStage === 'COMPLETED' ? 'COMPLETED' : 'OFFLINE'}</b></span>
            <span>Throughput: <b style={{ color: '#0EA5E9' }}>{processingFps ? `${processingFps.toFixed(1)} FPS` : 'N/A'}</b></span>
            <span>Inference: <b style={{ color: '#10B981' }}>{telemetry?.inference_time_ms ? `${telemetry.inference_time_ms} ms/frame` : processingFps ? `${(1000 / processingFps).toFixed(0)} ms/frame` : 'N/A'}</b></span>
            <span>Model Hash: <b style={{ fontFamily: 'monospace', color: '#64748B' }}>c789161e...</b></span>
          </div>
        </div>


        {activeModule === 'examination' ? (
          <>
            <PrimaryFindingPanel
              primaryFinding={primaryExamFinding || examEvents.find(e => e.primary_finding) || (examEvents.find(e => e.priority === 'RED') || null)}
              onReview={handleReviewEvent}
              onSeekTime={(tSec) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = tSec;
                  const vFps = telemetry?.video_fps || 30.0;
                  setFrame(Math.floor(tSec * vFps));
                }
              }}
              jobId={activeJobId || undefined}
            />
            {/* Gemini Multimodal Contextual Intelligence */}
            <GeminiContextualPanel
              jobId={activeJobId || null}
              geminiStatus={geminiStatus}
              geminiAnalysis={geminiExamAnalysis}
              isAnalyzing={isAnalyzingGemini}
              onAnalyze={handleTriggerGeminiAnalysis}
              onSeekTime={(tSec) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = tSec;
                  const vFps = telemetry?.video_fps || 30.0;
                  setFrame(Math.floor(tSec * vFps));
                }
              }}
              errorDetail={geminiError}
            />
            <ExaminationOverviewCard
              totalPersons={examTracks.length || liveTrackCount}
              totalEvents={examEvents.length}
              pendingReviews={examEvents.filter(e => e.review_status === 'PENDING').length}
              acceptedReviews={examEvents.filter(e => e.review_status === 'ACCEPTED').length}
              activeZones={examZones.length}
              pipelineStatus={pipelineStage === 'LIVE_INFERENCE' ? 'STREAMING INFERENCE' : pipelineStage === 'COMPLETED' ? 'ANALYSIS FINALIZED' : pipelineStage}
              studentCount={examTracks.filter(t => t.assigned_role === 'STUDENT').length}
              staffCount={examTracks.filter(t => ['INVIGILATOR', 'EXAM_COORDINATOR', 'AUTHORIZED_STAFF'].includes(t.assigned_role || '')).length}
              unknownCount={examTracks.filter(t => !t.assigned_role || t.assigned_role === 'UNKNOWN').length}
            />
            <ExaminationEventsTable
              events={examEvents}
              onReview={handleReviewEvent}
              onInspectPerson={(pId) => { setSelectedPersonId(pId); setSelTrack(pId); }}
              selectedPersonId={selectedPersonId}
              onSeekTime={(tSec) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = tSec;
                  const vFps = telemetry?.video_fps || 30.0;
                  setFrame(Math.floor(tSec * vFps));
                }
              }}
            />
          </>
        ) : (
          <>
            {/* Traffic Intelligence Overview */}
            <div className="card" style={{ marginTop: '16px' }}>
              <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Traffic Intelligence Overview</span>
                <span className="badge" style={{
                  background: (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'SEVERE' ? '#FEE2E2' : (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'CONGESTED' ? '#FEF3C7' : '#D1FAE5',
                  color: (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'SEVERE' ? '#991B1B' : (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'CONGESTED' ? '#92400E' : '#065F46',
                  fontWeight: 700, fontSize: '12px'
                }}>
                  CONGESTION: {telemetry?.traffic_intelligence?.congestion?.congestion_state || (pipelineStage === 'LIVE_INFERENCE' ? 'ANALYZING...' : 'FREE_FLOW')}
                </span>
              </h3>


          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', margin: '12px 0' }}>
            {[
              { label: 'VEHICLES OBSERVED', value: telemetry?.traffic_intelligence?.counting?.total_vehicle_count ?? (pipelineStage === 'LIVE_INFERENCE' ? liveTrackCount : 0), unit: '', color: '#1E293B' },
              { label: 'CURRENT OCCUPANCY', value: telemetry?.traffic_intelligence?.density?.current_road_occupancy ?? 0, unit: 'veh', color: '#4F46E5' },
              { label: 'MEAN DENSITY', value: telemetry?.traffic_intelligence?.density?.mean_density_veh_km ?? 0, unit: 'veh/km', color: '#0EA5E9' },
              { label: 'FLOW RATE', value: telemetry?.traffic_intelligence?.flow?.flow_rate_vph ?? 0, unit: 'veh/h', color: '#6366F1' },
              { label: 'AVERAGE SPEED', value: telemetry?.traffic_intelligence?.congestion?.avg_speed_kmh ?? 0, unit: 'km/h', color: '#10B981' },
              { label: 'INFERENCE FPS', value: processingFps?.toFixed(1) ?? (telemetry?.average_fps?.toFixed(1) || 'N/A'), unit: '', color: '#F59E0B' },
            ].map((metric, i) => (
              <div key={i} style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>{metric.label}</div>
                <div style={{ fontSize: '20px', fontWeight: 800, color: metric.color, marginTop: '2px' }}>
                  {metric.value} <small style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>{metric.unit}</small>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Phase 3: Speed Limit & Violations Configuration Panel */}
        <div className="card" style={{ marginTop: '16px', background: '#0F172A', color: '#F8FAFC', border: '1px solid #1E293B' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ margin: 0, color: '#F8FAFC', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span>⚡ Speed Limit & Violations</span>
              </h3>
              {isConfigUnsaved ? (
                <span className="badge" style={{ background: '#FEF3C7', color: '#B45309', fontWeight: 700, fontSize: '10px' }}>
                  ● UNSAVED CHANGES
                </span>
              ) : (
                <span className="badge" style={{ background: '#D1FAE5', color: '#065F46', fontWeight: 700, fontSize: '10px' }}>
                  ✓ SAVED & ACTIVE
                </span>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', color: '#94A3B8' }}>
              <span>Active Limit: <b style={{ color: '#38BDF8' }}>{appliedSpeedLimit} km/h</b></span>
              <span>Tolerance: <b style={{ color: '#FCD34D' }}>+{appliedTolerance} km/h</b></span>
              <span>Effective Threshold: <b style={{ color: '#F87171' }}>{appliedSpeedLimit + appliedTolerance} km/h</b></span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '12px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label style={{ fontSize: '11px', color: '#94A3B8', fontWeight: 600 }}>SPEED LIMIT:</label>
              <input
                type="number"
                min="1"
                max="300"
                value={speedLimitInput}
                onChange={e => {
                  const val = Number(e.target.value);
                  setSpeedLimitInput(val);
                  setIsConfigUnsaved(true);
                  setSpeedConfigError(null);
                }}
                style={{ width: '80px', padding: '4px 8px', borderRadius: '4px', border: '1px solid #334155', background: '#1E293B', color: '#F8FAFC', fontSize: '12px', fontWeight: 700 }}
              />
              <span style={{ fontSize: '11px', color: '#64748B' }}>km/h</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label style={{ fontSize: '11px', color: '#94A3B8', fontWeight: 600 }}>WARNING TOLERANCE:</label>
              <input
                type="number"
                min="0"
                max="50"
                value={warningToleranceInput}
                onChange={e => {
                  const val = Number(e.target.value);
                  setWarningToleranceInput(val);
                  setIsConfigUnsaved(true);
                  setSpeedConfigError(null);
                }}
                style={{ width: '70px', padding: '4px 8px', borderRadius: '4px', border: '1px solid #334155', background: '#1E293B', color: '#F8FAFC', fontSize: '12px', fontWeight: 700 }}
              />
              <span style={{ fontSize: '11px', color: '#64748B' }}>km/h</span>
            </div>

            <button
              onClick={handleApplySpeedLimit}
              style={{
                background: '#2563EB',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '4px',
                padding: '6px 14px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'background 0.15s'
              }}
            >
              Apply Speed Limit
            </button>

            <div style={{ fontSize: '11px', color: '#64748B', marginLeft: 'auto' }}>
              Effective Threshold = Limit ({speedLimitInput} km/h) + Tolerance ({warningToleranceInput} km/h) = <span style={{ color: '#F87171', fontWeight: 700 }}>{speedLimitInput + warningToleranceInput} km/h</span>
            </div>
          </div>

          {speedConfigError && (
            <div style={{ marginTop: '8px', color: '#F87171', fontSize: '11px', fontWeight: 600 }}>
              ⚠ {speedConfigError}
            </div>
          )}
        </div>

        {/* Phase 4: Violations Queue */}
        <div className="card" style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
            <div>
              <h3 style={{ margin: 0 }}>Violation Queue</h3>
              <p style={{ margin: '2px 0 0 0', fontSize: '11px', color: '#64748B' }}>
                Candidate violations awaiting human review · Legal notice: Not an automated enforcement citation.
              </p>
            </div>
            <span className="badge" style={{ background: '#FEE2E2', color: '#991B1B', fontWeight: 700 }}>
              {violations.length} CANDIDATE VIOLATIONS
            </span>
          </div>

          <div className="viol" style={{ marginTop: '10px', maxHeight: '280px', overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #E2E8F0', color: '#64748B', textAlign: 'left', height: '28px' }}>
                  <th style={{ padding: '6px 8px' }}>JOB/VID</th>
                  <th style={{ padding: '6px 8px' }}>TRACK ID</th>
                  <th style={{ padding: '6px 8px' }}>TIMESTAMP</th>
                  <th style={{ padding: '6px 8px' }}>MEASURED SPEED</th>
                  <th style={{ padding: '6px 8px' }}>SPEED LIMIT</th>
                  <th style={{ padding: '6px 8px' }}>TOLERANCE</th>
                  <th style={{ padding: '6px 8px' }}>OVER THRESHOLD</th>
                  <th style={{ padding: '6px 8px' }}>EVIDENCE</th>
                  <th style={{ padding: '6px 8px' }}>REVIEW STATUS</th>
                  <th style={{ padding: '6px 8px', textAlign: 'right' }}>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {violations.length === 0 ? (
                  <tr>
                    <td colSpan={10} style={{ textAlign: 'center', padding: '24px', color: '#94A3B8' }}>
                      No speed violations detected for active threshold ({appliedSpeedLimit + appliedTolerance} km/h).
                    </td>
                  </tr>
                ) : (
                  violations.map((e, idx) => {
                    const timeSec = typeof e.time === 'number' ? e.time : typeof e.t === 'number' ? e.t : 0;
                    const trackId = e.trackId ?? e.id;
                    const speedVal = e.speedKmh ?? e.v ?? 0;
                    const limitVal = e.limitKmh ?? appliedSpeedLimit;
                    const tolVal = e.appliedToleranceKmh ?? appliedTolerance;
                    const effThresh = e.effectiveThresholdKmh ?? (limitVal + tolVal);
                    const overThresh = e.amountOverThresholdKmh ?? Math.max(0, speedVal - effThresh);
                    const hasEvidence = e.evidenceAvailable && (e.fullFrameUrl || e.cropFrameUrl);

                    return (
                      <tr key={idx} style={{ borderBottom: '1px solid #F1F5F9' }}>
                        <td style={{ padding: '6px 8px', color: '#64748B' }}>
                          #{e.jobId || activeJobId || 'JOB'}
                        </td>
                        <td style={{ padding: '6px 8px', fontWeight: 700, color: '#1E293B' }}>
                          #{trackId}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {timeSec.toFixed(1)}s
                        </td>
                        <td style={{ padding: '6px 8px', color: '#EF4444', fontWeight: 700 }}>
                          {typeof speedVal === 'number' ? speedVal.toFixed(1) : speedVal} km/h
                        </td>
                        <td style={{ padding: '6px 8px', color: '#334155' }}>
                          {limitVal} km/h
                        </td>
                        <td style={{ padding: '6px 8px', color: '#64748B' }}>
                          +{tolVal} km/h
                        </td>
                        <td style={{ padding: '6px 8px', color: '#DC2626', fontWeight: 700 }}>
                          +{typeof overThresh === 'number' ? overThresh.toFixed(1) : overThresh} km/h
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {hasEvidence ? (
                            <button
                              className="b g"
                              style={{ padding: '2px 6px', fontSize: '10px' }}
                              onClick={() => setActiveEvidenceModal(e)}
                            >
                              📷 View Frame
                            </button>
                          ) : (
                            <span style={{ fontSize: '10px', color: '#94A3B8' }} title={e.evidenceReason || 'No crop frame stored'}>
                              Unavailable
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          <span className="badge" style={{
                            background: (e.status === 'ACCEPTED' ? '#D1FAE5' : e.status === 'REJECTED' ? '#FEE2E2' : '#FEF3C7'),
                            color: (e.status === 'ACCEPTED' ? '#065F46' : e.status === 'REJECTED' ? '#991B1B' : '#92400E'),
                            fontSize: '10px',
                            fontWeight: 700
                          }}>
                            {e.status || 'PENDING'}
                          </span>
                        </td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                          <button
                            className="b g"
                            style={{ padding: '2px 8px', fontSize: '11px' }}
                            onClick={() => {
                              handleInspectTrack(trackId, Math.max(0, Math.round(timeSec * (telemetry?.video_fps || FPS))));
                            }}
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Phase 5: Synchronized Vehicle Speed Intelligence Table with Filter Tabs & Accurate Pagination */}
        {(() => {
          const allTableData = getVehicleTableData();
          const sustainedCount = allTableData.filter((r: any) => r.isSustained).length;
          const inFrameCount = allTableData.filter((r: any) => r.track_status === 'IN FRAME' || r.track_status === 'ACTIVE').length;
          const calibratedCount = allTableData.filter((r: any) => r.isCalibrated).length;

          const filteredRows = allTableData.filter((row: any) => {
            if (tableFilter === 'SUSTAINED') return row.isSustained;
            if (tableFilter === 'IN_FRAME') return row.track_status === 'IN FRAME' || row.track_status === 'ACTIVE';
            return true;
          });

          const totalTablePages = Math.max(1, Math.ceil(filteredRows.length / TABLE_PAGE_SIZE));
          const currentTablePage = Math.min(tablePage, totalTablePages);
          const paginatedRows = filteredRows.slice((currentTablePage - 1) * TABLE_PAGE_SIZE, currentTablePage * TABLE_PAGE_SIZE);

          return (
            <div className="card" style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                <h3 style={{ margin: 0 }}>Vehicle Speed Intelligence</h3>
                <span className="badge" style={{ background: '#0F172A', color: '#10B981', fontWeight: 700 }}>
                  {filteredRows.length} TRACKS ({tableFilter}) · PAGE {currentTablePage}/{totalTablePages}
                </span>
              </div>

              {/* Phase 5 Track Aggregation Category Filters */}
              <div style={{ display: 'flex', gap: '8px', margin: '10px 0', alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 700 }}>CATEGORY:</span>
                <button
                  className={`tab-pill ${tableFilter === 'ALL' ? 'active' : 'inactive'}`}
                  style={{
                    padding: '3px 10px', fontSize: '11px', fontWeight: 700, borderRadius: '4px', border: 'none', cursor: 'pointer',
                    background: tableFilter === 'ALL' ? '#2563EB' : '#1E293B', color: '#FFF'
                  }}
                  onClick={() => { setTableFilter('ALL'); setTablePage(1); }}
                >
                  All Unique Tracks ({allTableData.length})
                </button>
                <button
                  className={`tab-pill ${tableFilter === 'SUSTAINED' ? 'active' : 'inactive'}`}
                  style={{
                    padding: '3px 10px', fontSize: '11px', fontWeight: 700, borderRadius: '4px', border: 'none', cursor: 'pointer',
                    background: tableFilter === 'SUSTAINED' ? '#2563EB' : '#1E293B', color: '#FFF'
                  }}
                  onClick={() => { setTableFilter('SUSTAINED'); setTablePage(1); }}
                >
                  Sustained Tracks (≥5f) ({sustainedCount})
                </button>
                <button
                  className={`tab-pill ${tableFilter === 'IN_FRAME' ? 'active' : 'inactive'}`}
                  style={{
                    padding: '3px 10px', fontSize: '11px', fontWeight: 700, borderRadius: '4px', border: 'none', cursor: 'pointer',
                    background: tableFilter === 'IN_FRAME' ? '#2563EB' : '#1E293B', color: '#FFF'
                  }}
                  onClick={() => { setTableFilter('IN_FRAME'); setTablePage(1); }}
                >
                  In Frame ({inFrameCount})
                </button>
                <span style={{ marginLeft: 'auto', fontSize: '11px', color: '#64748B' }}>
                  Calibrated Metric Speeds: <b style={{ color: '#10B981' }}>{calibratedCount}</b> / {allTableData.length}
                </span>
              </div>

              <div style={{ maxHeight: '320px', overflowY: 'auto', marginTop: '6px' }}>
                <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #E2E8F0', color: '#64748B', textAlign: 'left', height: '32px' }}>
                      <th style={{ padding: '6px 8px' }}>TRACK ID</th>
                      <th style={{ padding: '6px 8px' }}>CLASS</th>
                      <th style={{ padding: '6px 8px' }}>CURRENT SPEED</th>
                      <th style={{ padding: '6px 8px' }}>AVG SPEED</th>
                      <th style={{ padding: '6px 8px' }}>MAX SPEED</th>
                      <th style={{ padding: '6px 8px' }}>TRACK STATUS</th>
                      <th style={{ padding: '6px 8px' }}>CALIBRATION</th>
                      <th style={{ padding: '6px 8px' }}>VALIDITY</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedRows.length === 0 ? (
                      <tr>
                        <td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: '#94A3B8' }}>
                          No tracked vehicles matching selected filter.
                        </td>
                      </tr>
                    ) : (
                      paginatedRows.map((row: any) => {
                        const isSelected = selTrack === row.track_id;
                        const hasValidSpeed = row.isCalibrated && row.current_speed_raw !== null && row.current_speed_raw > 0;
                        const isSpeeding = hasValidSpeed && row.current_speed_raw > (appliedSpeedLimit + appliedTolerance);
                        const isUncalibrated = row.measurement_state === 'UNCALIBRATED_MOTION';

                        return (
                          <tr
                            key={row.track_id}
                            onClick={() => handleInspectTrack(row.track_id)}
                            style={{
                              borderBottom: '1px solid #F1F5F9',
                              background: isSelected ? '#FEF3C7' : 'transparent',
                              cursor: 'pointer',
                              transition: 'background 0.15s'
                            }}
                          >
                            <td style={{ padding: '8px', fontWeight: 700, color: '#1E293B' }}>
                              #{row.track_id}
                            </td>
                            <td style={{ padding: '8px' }}>
                              <span className="badge" style={{ fontSize: '10px' }}>{row.vehicle_class}</span>
                            </td>
                            <td style={{
                              padding: '8px',
                              fontWeight: 700,
                              color: isSpeeding ? '#EF4444' : hasValidSpeed ? '#10B981' : isUncalibrated ? '#0EA5E9' : '#64748B'
                            }}>
                              {row.current_speed}
                            </td>
                            <td style={{ padding: '8px', color: '#334155' }}>
                              {row.avg_speed}
                            </td>
                            <td style={{ padding: '8px', color: '#334155' }}>
                              {row.max_speed}
                            </td>
                            <td style={{ padding: '8px' }}>
                              <span style={{
                                fontSize: '10px',
                                fontWeight: 700,
                                padding: '2px 6px',
                                borderRadius: '4px',
                                background: row.track_status === 'ACTIVE' || row.track_status === 'IN FRAME' ? '#DCFCE7' : row.track_status === 'SUSTAINED' ? '#E0E7FF' : '#F1F5F9',
                                color: row.track_status === 'ACTIVE' || row.track_status === 'IN FRAME' ? '#166534' : row.track_status === 'SUSTAINED' ? '#3730A3' : '#475569'
                              }}>
                                {row.track_status}
                              </span>
                            </td>
                            <td style={{ padding: '8px' }}>
                              <span style={{
                                fontSize: '10px',
                                fontWeight: 600,
                                color: row.calibration_status === 'CALIBRATED' || row.calibration_status === 'SYNTHETIC' ? '#2563EB' : '#D97706'
                              }}>
                                {row.calibration_status}
                              </span>
                            </td>
                            <td style={{ padding: '8px' }}>
                              <span style={{
                                fontSize: '10px',
                                fontWeight: 600,
                                color: row.validity === 'VALID' ? '#10B981' : row.validity === 'UNCALIBRATED' ? '#0EA5E9' : '#D97706'
                              }}>
                                {row.validity}
                              </span>
                            </td>
                            <td style={{ padding: '8px', textAlign: 'right' }}>
                              <button
                                className="b g"
                                style={{ padding: '2px 8px', fontSize: '11px' }}
                                onClick={(ev) => {
                                  ev.stopPropagation();
                                  handleInspectTrack(row.track_id);
                                }}
                              >
                                Inspect
                              </button>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>

              {/* Scalability Pagination Controls (Requirement #6) */}
              {filteredRows.length > 0 && (
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '8px 12px',
                  borderTop: '1px solid #E2E8F0',
                  fontSize: '11px',
                  color: '#64748B',
                  marginTop: '6px'
                }}>
                  <div>
                    Showing <b>{(currentTablePage - 1) * TABLE_PAGE_SIZE + 1}</b>–<b>{Math.min(currentTablePage * TABLE_PAGE_SIZE, filteredRows.length)}</b> of <b>{filteredRows.length}</b> vehicles
                  </div>
                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                    <button
                      className="b g"
                      style={{ padding: '3px 8px', fontSize: '11px' }}
                      disabled={currentTablePage <= 1}
                      onClick={() => setTablePage(p => Math.max(1, p - 1))}
                    >
                      ◀ Prev
                    </button>
                    <span>Page <b>{currentTablePage}</b> of <b>{totalTablePages}</b></span>
                    <button
                      className="b g"
                      style={{ padding: '3px 8px', fontSize: '11px' }}
                      disabled={currentTablePage >= totalTablePages}
                      onClick={() => setTablePage(p => Math.min(totalTablePages, p + 1))}
                    >
                      Next ▶
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })()}
        </>
      )}

      </div>

      <div>
        <div className="card" style={{ background: '#0F172A', border: '1px solid #1E293B', color: '#F8FAFC' }}>
          {activeModule === 'examination' ? (
            <ExaminationSidebarInspector
              events={examEvents}
              selectedPersonId={selectedPersonId}
              personTracks={examTracks}
              zones={examZones}
              onReview={handleReviewEvent}
              onSeekTime={(tSec) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = tSec;
                  const vFps = telemetry?.video_fps || 30.0;
                  setFrame(Math.floor(tSec * vFps));
                }
              }}
            />
          ) : (
            <>
              <div style={{ display: 'flex', gap: 6, marginBottom: 12, borderBottom: '1px solid #1E293B', paddingBottom: 8 }}>
                <button
                  className={`tab-pill ${sidebarTab === 'log' ? 'active' : 'inactive'}`}
                  onClick={() => setSidebarTab('log')}
                >
                  LIVE DETECTION LOG
                </button>
                <button
                  className={`tab-pill ${sidebarTab === 'inspector' ? 'active' : 'inactive'}`}
                  onClick={() => setSidebarTab('inspector')}
                >
                  TRACK INSPECTOR {selTrack !== null && `(#${selTrack})`}
                </button>
              </div>
              {sidebarTab === 'log' ? renderLiveDetectionLog() : renderInspector()}
            </>
          )}
        </div>
      </div>

      {/* Evidence Viewer Modal */}
      {activeEvidenceModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 99999
        }} onClick={() => setActiveEvidenceModal(null)}>
          <div style={{ background: '#1E293B', padding: '20px', borderRadius: '8px', maxWidth: '680px', width: '92%', color: '#F8FAFC', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.5)' }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #334155', paddingBottom: '10px' }}>
              <h4 style={{ margin: 0, fontSize: '15px', color: '#F8FAFC' }}>
                📸 Candidate Violation Evidence · Track #{activeEvidenceModal.trackId ?? activeEvidenceModal.id}
              </h4>
              <button onClick={() => setActiveEvidenceModal(null)} style={{ background: 'none', border: 'none', color: '#94A3B8', fontSize: '18px', cursor: 'pointer' }}>✕</button>
            </div>
            <div style={{ margin: '12px 0', fontSize: '12px', color: '#CBD5E1', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
              <div>Measured Speed: <b style={{ color: '#EF4444' }}>{activeEvidenceModal.speedKmh ?? activeEvidenceModal.v} km/h</b></div>
              <div>Speed Limit: <b style={{ color: '#38BDF8' }}>{activeEvidenceModal.limitKmh} km/h</b></div>
              <div>Warning Tolerance: <b>+{activeEvidenceModal.appliedToleranceKmh || 0} km/h</b></div>
              <div>Over Threshold: <b style={{ color: '#F87171' }}>+{activeEvidenceModal.amountOverThresholdKmh || 0} km/h</b></div>
              <div>Review Status: <span className="badge" style={{ background: '#FEF3C7', color: '#92400E' }}>{activeEvidenceModal.status || 'PENDING'}</span></div>
              <div>Timestamp: <b>{typeof activeEvidenceModal.time === 'number' ? activeEvidenceModal.time.toFixed(1) + 's' : activeEvidenceModal.time}</b></div>
            </div>
            {activeEvidenceModal.fullFrameUrl && (
              <div style={{ marginTop: '10px', textAlign: 'center' }}>
                <img
                  src={apiUrl(activeEvidenceModal.fullFrameUrl)}
                  alt="Candidate Violation Frame"
                  style={{ maxWidth: '100%', maxHeight: '380px', borderRadius: '6px', border: '1px solid #334155' }}
                />
              </div>
            )}
            <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '10px', color: '#94A3B8' }}>
                * Candidate violation requiring human analyst review. Not a legal citation.
              </span>
              <button className="b g" onClick={() => setActiveEvidenceModal(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
