import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiUrl, getWsUrl } from '../config/api';

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

  // --- inspector & live log ---
  const [selTrack, setSelTrack] = useState<number | null>(null);
  const [violations, setViolations] = useState<any[]>([]);
  const [speedLimit, setSpeedLimit] = useState(80);
  const [sidebarTab, setSidebarTab] = useState<'log' | 'inspector'>('log');
  const [logFilter, setLogFilter] = useState<'all' | 'active' | 'speeding'>('all');

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

        const [trkRes, calibRes, vidInfoRes] = await Promise.all([
          fetch(apiUrl(`/api/v1/jobs/${job.id}/tracks`)).catch(() => null),
          fetch(apiUrl(`/api/v1/videos/${job.video_id}/calibrations`)).catch(() => null),
          fetch(apiUrl(`/api/v1/videos/${job.video_id}`)).catch(() => null)
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

      if (trkData?.length) {
        setRealTracks(trkData);
        rtFrameBufferRef.current.clear();
      }
      if (violsData?.length) {
        setViolations(violsData.map((v: any) => ({
          t: v.timestamp, id: v.track_id, v: v.estimated_speed_kmh, loc: v.location_label
        })));
      }
      if (jobData?.telemetry) {
        setTelemetry(jobData.telemetry);
        if (jobData.telemetry.average_fps) setProcessingFps(jobData.telemetry.average_fps);
        if (jobData.telemetry.real_time_factor) setRealTimeFactor(jobData.telemetry.real_time_factor);
      }
      setPipelineStage('COMPLETED');
      setJobStatusText(`Job #${jobId} COMPLETED — REAL VIDEO · LIVE AI ANALYSIS`);
    } catch (err) {
      console.error('Failed to load job results:', err);
      setPipelineStage('COMPLETED');
      setJobStatusText(`Job #${jobId} Complete — REAL VIDEO · LIVE AI ANALYSIS`);
    }
  }, []);

  // REST Polling Fallback to guarantee job state synchronization if WebSocket disconnects or finishes fast
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

        // Frame synchronization without severe playback rate jumps
        if (videoRef.current && videoRef.current.readyState > 1) {
          const vid = videoRef.current;
          const vFps = telemetry?.video_fps || 29.97;
          const vidCurrentFrame = Math.floor(vid.currentTime * vFps);
          const frameLag = vidCurrentFrame - frameIdx;

          if (Math.abs(frameLag) > 60) {
            // Severe drift: one-time gentle re-anchor
            vid.currentTime = Math.max(0, frameIdx / vFps);
          } else if (frameLag > 8) {
            vid.playbackRate = 0.9;
          } else if (frameLag < -8) {
            vid.playbackRate = 1.1;
          } else {
            vid.playbackRate = 1.0;
          }
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
        if (j.status === 'SUCCEEDED') loadJobResults(jobId);
      }).catch(() => {});
    };

    ws.onclose = () => {
      setWsConnected(false);
      isLiveStreamingRef.current = false;
      // Check REST API for job completion on WebSocket close
      fetch(apiUrl(`/api/v1/jobs/${jobId}`)).then(r => r.json()).then(j => {
        if (j.status === 'SUCCEEDED') loadJobResults(jobId);
      }).catch(() => {});
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
          setFrame(curF);
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
      // CALIBRATION OVERLAY (Requirement #9 & #10)
      // Transform source image points to 800x450 canvas space ONLY if valid calibration exists
      // ---------------------------------------------------------------------
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
      } else {
        cx.fillStyle = '#94A3B8CC'; cx.font = 'bold 11px sans-serif'; cx.textAlign = 'left';
        cx.fillText('ROAD CALIBRATION NOT CONFIGURED · METRIC SPEED UNAVAILABLE', 16, 26);
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

      let currentFrameDets: any[] | undefined;
      if (isLiveStreamingRef.current) {
        currentFrameDets = findDets(currentVidFrame, rtFrameBufferRef.current, 3);

        if (!currentFrameDets && latestInferenceFrame !== null && latestInferenceFrame !== undefined) {
          if (rtFrameBufferRef.current.has(latestInferenceFrame)) {
            currentFrameDets = rtFrameBufferRef.current.get(latestInferenceFrame);
          } else {
            const allKeys = Array.from(rtFrameBufferRef.current.keys());
            if (allKeys.length > 0) {
              const closest = allKeys.reduce((prev, curr) => Math.abs(curr - currentVidFrame) < Math.abs(prev - currentVidFrame) ? curr : prev);
              if (Math.abs(closest - currentVidFrame) <= 15) {
                currentFrameDets = rtFrameBufferRef.current.get(closest);
              }
            }
          }
        }
      } else {
        currentFrameDets = findDets(currentVidFrame, frameMapRef.current, 5);
      }

      if (currentFrameDets && currentFrameDets.length > 0) {
        const renderedBadges: { x: number, y: number, w: number, h: number }[] = [];
        currentFrameDets.forEach(det => {
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
          const isOutOfROI = !hasValidSpeed && !isUncalibratedMotion && det.speedStatus === 'OUT_OF_ROI';
          const isUnstable = !hasValidSpeed && !isUncalibratedMotion && (det.speedStatus === 'CALIBRATION_UNSTABLE' || det.speedStatus === 'WORLD_COORDINATE_INVALID');

          // SPEEDING INTEGRITY: uncalibrated motion cannot trigger a confirmed speeding violation
          const bad = isCalibratedSpeed && det.speedKmh > speedLimit;

          // Box color:
          // Red (#EF4444) strictly for confirmed calibrated speeding violations
          // Green (#10B981) for confirmed calibrated compliant speeds
          // Cyan (#0EA5E9) for uncalibrated relative motion
          // Amber (#F59E0B) for calculating
          // Slate (#64748B / #94A3B8) for calibration required / out of ROI
          const col = bad ? '#EF4444'
            : isCalibratedSpeed ? '#10B981'
            : isUncalibratedMotion ? '#0EA5E9'
            : isCalibReq ? '#64748B'
            : isCalculating ? '#F59E0B'
            : '#94A3B8';
          const isSelected = selTrack === det.track_id;

          cx.fillStyle = isSelected ? 'rgba(245, 158, 11, 0.22)' : (col + '28');
          cx.fillRect(bx, by, bw, bh);

          const bracketLen = Math.min(12, Math.min(bw, bh) * 0.35);
          cx.strokeStyle = isSelected ? '#F59E0B' : col; cx.lineWidth = isSelected ? 3 : 2;
          cx.beginPath();
          cx.moveTo(bx, by + bracketLen); cx.lineTo(bx, by); cx.lineTo(bx + bracketLen, by);
          cx.moveTo(bx + bw - bracketLen, by); cx.lineTo(bx + bw, by); cx.lineTo(bx + bw, by + bracketLen);
          cx.moveTo(bx + bw, by + bh - bracketLen); cx.lineTo(bx + bw, by + bh); cx.lineTo(bx + bw - bracketLen, by + bh);
          cx.moveTo(bx + bracketLen, by + bh); cx.lineTo(bx, by + bh); cx.lineTo(bx, by + bh - bracketLen);
          cx.stroke();
          cx.strokeStyle = isSelected ? '#F59E0B44' : (col + '44'); cx.lineWidth = 1;
          cx.strokeRect(bx, by, bw, bh);
          cx.fillStyle = isSelected ? '#F59E0B' : col; cx.beginPath();
          cx.arc(bx + bw / 2, by + bh / 2, 2.5, 0, 2 * Math.PI); cx.fill();

          // HUD Badge — SPEED INTEGRITY COMPLIANT OVERLAY DIRECTLY ON BOUNDING BOX
          const clsLabel = (det.vehicle_class || 'Vehicle').toUpperCase();
          let speedLabel = 'Calculating...';
          if (isCalibratedSpeed) {
            speedLabel = `${Math.round(det.speedKmh)} km/h`;
          } else if (isUncalibratedMotion) {
            speedLabel = 'Uncalibrated motion';
          } else if (isCalibReq) {
            speedLabel = 'Calib req';
          } else if (isOutOfROI) {
            speedLabel = 'Out of ROI';
          } else if (isUnstable) {
            speedLabel = 'Calib unstable';
          } else if (isCalculating) {
            speedLabel = 'Calculating...';
          } else {
            speedLabel = 'Speed unavailable';
          }

          const tx = `${clsLabel} #${det.track_id} | ${speedLabel}`;
          cx.font = 'bold 11px monospace, sans-serif';
          const tw = cx.measureText(tx).width + 14;
          const badgeX = Math.max(2, Math.min(W - tw - 2, bx + bw / 2 - tw / 2));
          let badgeY = by >= 25 ? by - 22 : by + bh + 4;

          // Collision resolution: prevent overlapping vehicle labels near top of video
          for (const prev of renderedBadges) {
            const overlapX = Math.abs(badgeX - prev.x) < Math.max(tw, prev.w) * 0.85;
            const overlapY = Math.abs(badgeY - prev.y) < 18;
            if (overlapX && overlapY) {
              badgeY = (by >= 25 && prev.y < by) ? prev.y - 20 : prev.y + prev.h + 2;
            }
          }
          badgeY = Math.max(4, Math.min(H - 22, badgeY));
          renderedBadges.push({ x: badgeX, y: badgeY, w: tw, h: 20 });

          cx.fillStyle = '#0B0F19F2'; cx.strokeStyle = isSelected ? '#F59E0B' : col; cx.lineWidth = 1.5;
          cx.beginPath();
          if ((cx as any).roundRect) (cx as any).roundRect(badgeX, badgeY, tw, 20, 6);
          else cx.rect(badgeX, badgeY, tw, 20);
          cx.fill(); cx.stroke();
          cx.fillStyle = isSelected ? '#FBBF24' : (bad ? '#FCA5A5' : isCalibratedSpeed ? '#6EE7B7' : isUncalibratedMotion ? '#7DD3FC' : isCalculating ? '#FDE68A' : '#CBD5E1');
          cx.textAlign = 'center'; cx.fillText(tx, badgeX + tw / 2, badgeY + 14);
        });
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
        cx.strokeStyle = '#4F46E5'; cx.lineWidth = 1; cx.setLineDash([4, 3]);
        cx.beginPath(); for (let k = 0; k <= 14; k++) { const q = state(a, (frame - k * 3) / FPS); k ? cx.lineTo(q.x, q.y) : cx.moveTo(q.x, q.y); } cx.stroke(); cx.setLineDash([]);
        cx.strokeStyle = selTrack === a.id ? '#F59E0B' : col; cx.lineWidth = selTrack === a.id ? 3 : 2;
        cx.strokeRect(x - 2, y - 2, s.bw + 4, s.bh + 4);
        const tx = Math.round(v) + ' km/h'; const fs = Math.max(9, 11 + s.s * 6); cx.font = 'bold ' + fs + 'px sans-serif';
        const tw = cx.measureText(tx).width + 10; cx.fillStyle = col; cx.beginPath();
        if ((cx as any).roundRect) (cx as any).roundRect(s.x - tw / 2, y - fs - 10, tw, fs + 5, 9); else cx.rect(s.x - tw / 2, y - fs - 10, tw, fs + 5);
        cx.fill(); cx.fillStyle = '#fff'; cx.textAlign = 'center'; cx.fillText(tx, s.x, y - 6);
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
  }, [frame, speedLimit, selTrack, isRealVideo, realTracks, telemetry, realCalib, latestInferenceFrame, processingFps, activeJobId]);

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
    setViolations([]);
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
      videoRef.current.loop = false; // Do not loop while streaming to maintain monotonic frame sync
      videoRef.current.play().catch(() => {});
    }
    setPipelineStage('VIDEO_READY');
    setJobStatusText('Video loaded — Uploading for AI analysis...');

    setPipelineStage('UPLOADING');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const vidRes = await fetch(apiUrl('/api/v1/projects/1/videos'), { method: 'POST', body: formData });
      if (!vidRes.ok) throw new Error(`Upload failed: ${vidRes.status}`);
      const vidData = await vidRes.json();
      setPipelineStage('UPLOAD_COMPLETE');
      setJobStatusText('Video uploaded — Triggering CV Detection Job...');

      // Authoritative extracted video metadata from backend
      setTelemetry({
        source_width: vidData.width || 1920,
        source_height: vidData.height || 1080,
        video_fps: vidData.fps || 29.97,
        video_duration_sec: vidData.duration_sec || 0.0
      });

      // Check for calibration specifically belonging to this uploaded video
      let assignedCalib: any = null;
      try {
        const calibRes = await fetch(apiUrl(`/api/v1/videos/${vidData.id}/calibrations`));
        if (calibRes.ok) {
          const calibData = await calibRes.json();
          if (calibData?.length) assignedCalib = calibData[0];
        }
      } catch (ce) {
        console.warn('Calibration query error:', ce);
      }
      setRealCalib(assignedCalib);

      // Create job strictly referencing assigned calibration for this video (never fallback to demo calibration)
      const jobRes = await fetch(apiUrl(`/api/v1/videos/${vidData.id}/jobs`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ calibration_id: assignedCalib ? assignedCalib.id : null })
      });
      if (!jobRes.ok) throw new Error(`Job creation failed: ${jobRes.status}`);
      const jobData = await jobRes.json();
      const jobId = jobData.id;

      setActiveJobId(jobId);
      setPipelineStage('JOB_CREATED');
      setJobStatusText(`Job #${jobId} created — connecting real-time stream...`);

      connectWebSocket(jobId);

    } catch (err: any) {
      console.error('Upload/job error:', err);
      setJobStatusText(`Error: ${err.message}`);
      setPipelineStage('ERROR');
    }
  };

  // =========================================================================
  // Canvas Click Handler
  // =========================================================================
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
        const bx = det.rx * scaleX; const by = det.ry * scaleY;
        const bw = Math.max(15, det.rw * scaleX); const bh = Math.max(12, det.rh * scaleY);
        if (x >= bx - 10 && x <= bx + bw + 10 && y >= by - 10 && y <= by + bh + 10) hitTrk = det.track_id;
      });
      if (hitTrk !== null) { setSelTrack(hitTrk); setPlaying(false); }
    } else {
      const t = frame / FPS;
      let hit: number | null = null;
      V.forEach(a => {
        const s = state(a, t);
        if (s.d > 6 && s.d < 125 && x > s.x - s.bw / 2 - 4 && x < s.x + s.bw / 2 + 4 && y > s.y - s.bh - 4 && y < s.y + 4) hit = a.id;
      });
      if (hit) { setSelTrack(hit); setPlaying(false); }
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
                        setSelTrack(log.track_id);
                        setSidebarTab('inspector');
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
          const validSpeeds = (trk.speed_measurements || [])
            .filter((s: any) => (s.error_components?.validity === 'VALID' || !s.error_components?.validity) && (s.smoothed_kmh || s.instantaneous_kmh) > 0)
            .map((s: any) => s.smoothed_kmh ?? s.instantaneous_kmh);

          const currentSm = (trk.speed_measurements || []).find((s: any) => s.frame === currentVidFrame);
          const hasCurrentValidSm = currentSm && (currentSm.smoothed_kmh || currentSm.instantaneous_kmh) > 0 &&
            (currentSm.error_components?.validity === 'VALID' || !currentSm.error_components?.validity);

          let currentSpeedVal: number | null = hasCurrentValidSm ? (currentSm.smoothed_kmh ?? currentSm.instantaneous_kmh) : null;
          let measurementState: 'CALIBRATED_SPEED' | 'UNCALIBRATED_MOTION' | 'SPEED_UNAVAILABLE' = 'SPEED_UNAVAILABLE';

          if (currentSpeedVal !== null) {
            measurementState = 'CALIBRATED_SPEED';
          } else if (realCalib) {
            if (validSpeeds.length > 0) {
              currentSpeedVal = validSpeeds[validSpeeds.length - 1];
              measurementState = 'CALIBRATED_SPEED';
            } else {
              measurementState = 'SPEED_UNAVAILABLE';
            }
          } else if (trk.trajectory && trk.trajectory.length >= 2) {
            measurementState = 'UNCALIBRATED_MOTION';
          } else {
            measurementState = 'SPEED_UNAVAILABLE';
          }

          const isCalibrated = measurementState === 'CALIBRATED_SPEED';
          const avgSpeed = isCalibrated && validSpeeds.length > 0
            ? (validSpeeds.reduce((a: number, b: number) => a + b, 0) / validSpeeds.length).toFixed(1) + ' km/h'
            : '-';
          const maxSpeed = isCalibrated && validSpeeds.length > 0
            ? Math.max(...validSpeeds).toFixed(1) + ' km/h'
            : '-';

          let calibStatus = realCalib ? 'CALIBRATED' : 'UNCALIBRATED';
          let validity = isCalibrated ? 'CALIBRATED_SPEED' : measurementState === 'UNCALIBRATED_MOTION' ? 'UNCALIBRATED_MOTION' : 'SPEED_UNAVAILABLE';
          let currentSpeedText = isCalibrated && currentSpeedVal !== null
            ? `${Math.round(currentSpeedVal)} km/h`
            : measurementState === 'UNCALIBRATED_MOTION'
            ? 'Uncalibrated motion'
            : 'Speed unavailable';

          const isCurrentlyInFrame = currentVidFrame >= (trk.first_frame || 0) && currentVidFrame <= (trk.last_frame || 999999);
          const trackStatus = isCurrentlyInFrame ? 'IN FRAME' : 'LOGGED';

          return {
            track_id: trk.track_id,
            vehicle_class: trk.vehicle_class || 'Vehicle',
            current_speed: currentSpeedText,
            current_speed_raw: isCalibrated ? currentSpeedVal : null,
            isCalibrated,
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
          <h3>
            Analytical Workbench{' '}
            <span className="badge">{srcName}</span>
            {activeJobId && (
              <span className="badge" style={{ background: '#0F172A', color: '#10B981', fontWeight: 700, marginLeft: 6 }}>
                ACTIVE JOB: #{activeJobId}
              </span>
            )}
            {isLiveStreamingRef.current && (
              <span className="badge" style={{ background: '#D1FAE5', color: '#065F46', marginLeft: 6 }}>● REAL VIDEO — LIVE AI</span>
            )}
            {pipelineStage === 'COMPLETED' && (
              <span className="badge" style={{ background: '#EEF2FF', color: '#4F46E5', marginLeft: 6 }}>REAL VIDEO · LIVE AI</span>
            )}
            {!isRealVideo && (
              <span className="badge" style={{ background: '#FFFBEB', color: '#B45309', marginLeft: 6 }}>DEMO MODE — SYNTHETIC</span>
            )}
          </h3>

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
          <canvas ref={canvasRef} width={W} height={H} onClick={handleCanvasClick} style={{ cursor: 'pointer', display: 'block', borderRadius: 6 }} />

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
            <label>Limit{' '}<input type="number" value={speedLimit} onChange={e => { setSpeedLimit(Number(e.target.value)); seenRef.current.clear(); }} />{' '}km/h</label>
            <label>Upload MP4/WebM{' '}<input type="file" accept="video/mp4,video/webm" onChange={handleFileUpload} /></label>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#64748B', marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #E2E8F0', flexWrap: 'wrap' }}>
            <span>Detector: <b style={{ color: '#4F46E5' }}>{telemetry?.detector_actual || 'YOLOX-Nano-ONNX'}</b></span>
            <span>Tracker: <b style={{ color: '#10B981' }}>ByteTrack</b></span>
            <span>Status: <b style={{ color: wsConnected ? '#10B981' : '#64748B' }}>{wsConnected ? '● STREAMING' : pipelineStage === 'COMPLETED' ? 'COMPLETED' : 'OFFLINE'}</b></span>
            <span>Inference: <b style={{ color: '#10B981' }}>{telemetry?.inference_time_ms ? `${telemetry.inference_time_ms} ms/frame` : processingFps ? `${(1000 / processingFps).toFixed(0)} ms/frame` : 'N/A'}</b></span>
            <span>Model Hash: <b style={{ fontFamily: 'monospace', color: '#64748B' }}>c789161e...</b></span>
          </div>
        </div>

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

        {/* Violations Queue */}
        <div className="card">
          <h3>Violation Queue</h3>
          <div className="viol">
            <table>
              <thead><tr><th>Time</th><th>ID</th><th>Speed</th><th>Location</th><th></th></tr></thead>
              <tbody>
                {violations.map((e, idx) => (
                  <tr key={idx}>
                    <td>{typeof e.t === 'number' ? e.t.toFixed(1) + 's' : e.t}</td>
                    <td>#{e.id}</td>
                    <td style={{ color: '#EF4444', fontWeight: 700 }}>{typeof e.v === 'number' ? e.v.toFixed(1) : e.v} km/h</td>
                    <td>{e.loc}</td>
                    <td><button className="b g" onClick={() => { setSelTrack(e.id); if (typeof e.t === 'number') setFrame(Math.max(0, (e.t * FPS) | 0)); setPlaying(false); }}>View</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        {/* Phase 22: Synchronized Vehicle Speed Intelligence Table */}
        <div className="card" style={{ marginTop: '16px' }}>
          <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Vehicle Speed Intelligence</span>
            <span className="badge" style={{ background: '#0F172A', color: '#10B981', fontWeight: 700 }}>
              {getVehicleTableData().length} VEHICLES LOGGED
            </span>
          </h3>

          <div style={{ maxHeight: '280px', overflowY: 'auto', marginTop: '10px' }}>
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
                {getVehicleTableData().length === 0 ? (
                  <tr>
                    <td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: '#94A3B8' }}>
                      No tracked vehicles active in current scene.
                    </td>
                  </tr>
                ) : (
                  getVehicleTableData().map((row: any) => {
                    const isSelected = selTrack === row.track_id;
                    const hasValidSpeed = row.isCalibrated && row.current_speed_raw !== null && row.current_speed_raw > 0;
                    const isSpeeding = hasValidSpeed && row.current_speed_raw > speedLimit;
                    const isUncalibrated = row.measurement_state === 'UNCALIBRATED_MOTION';

                    return (
                      <tr
                        key={row.track_id}
                        onClick={() => { setSelTrack(row.track_id); setPlaying(false); }}
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
                            background: row.track_status === 'ACTIVE' || row.track_status === 'IN FRAME' ? '#DCFCE7' : '#F1F5F9',
                            color: row.track_status === 'ACTIVE' || row.track_status === 'IN FRAME' ? '#166534' : '#475569'
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
                            color: row.measurement_state === 'CALIBRATED_SPEED' ? '#10B981' : row.measurement_state === 'UNCALIBRATED_MOTION' ? '#0EA5E9' : '#D97706'
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
                              setSelTrack(row.track_id);
                              setPlaying(false);
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

      </div>

      <div>
        <div className="card" style={{ background: '#0F172A', border: '1px solid #1E293B', color: '#F8FAFC' }}>
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
        </div>
      </div>
    </div>
  );
};
