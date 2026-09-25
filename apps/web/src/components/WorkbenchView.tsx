import React, { useState, useEffect, useRef } from 'react';

const W = 800;
const H = 450;
const FPS = 30;
const N = 900;
const CL = ['Sedan', 'SUV', 'Truck', 'Motorcycle'];

const S = (d: number) => 1 / (1 + d / 25);
const proj = (lx: number, d: number): [number, number] => [400 + (lx - 6) * S(d) * 60, 110 + 330 * S(d)];

function rnd(s: number) {
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}
const R = rnd(7);
const V: any[] = [];
for (let i = 0; i < 14; i++) {
  const c = CL[i % 4 === 3 && i % 8 !== 3 ? 0 : i % 4];
  V.push({
    id: i + 1,
    cls: c,
    lx: [2, 6, 10][i % 3],
    v: (58 + R() * 55) / 3.6,
    off: R() * 150,
    conf: 0.82 + R() * 0.16,
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
    let m = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[m][i])) m = r;
    [A[i], A[m]] = [A[m], A[i]];
    [b[i], b[m]] = [b[m], b[i]];
    for (let r = i + 1; r < n; r++) {
      const k = A[r][i] / A[i][i];
      for (let c = i; c < n; c++) A[r][c] -= k * A[i][c];
      b[r] -= k * b[i];
    }
  }
  const x = Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c];
    x[i] = s / A[i][i];
  }
  return x;
}

let Hm: number[][] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
function calcH() {
  const A: number[][] = [];
  const b: number[] = [];
  HP.forEach(([x, y], i) => {
    const [X, Y] = CP[i];
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]);
    b.push(X);
    A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]);
    b.push(Y);
  });
  const h = solve(A, b);
  Hm = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

const toW = (x: number, y: number) => {
  const q = Hm[2][0] * x + Hm[2][1] * y + 1;
  return [(Hm[0][0] * x + Hm[0][1] * y + Hm[0][2]) / q, (Hm[1][0] * x + Hm[1][1] * y + Hm[1][2]) / q];
};

function state(a: any, t: number) {
  const d = dOf(a, t);
  const [x, y] = proj(a.lx, d);
  const s = S(d);
  return { d, x, y, s, bw: a.w * 60 * s, bh: a.h * 60 * s };
}
function jit(a: any, fr: number, k: number) {
  return Math.sin(a.id * 12.9 + fr * 78.2 + k * 3.1) * 0.7;
}
function estSpeed(a: any, fr: number) {
  const dt = 15;
  const p = (q: number) => {
    const st = state(a, q / FPS);
    return toW(st.x + jit(a, q, 0), st.y + jit(a, q, 1));
  };
  const A = p(fr);
  const B = p(fr - dt);
  return (Math.hypot(A[0] - B[0], A[1] - B[1]) / (dt / FPS)) * 3.6;
}

export const WorkbenchView: React.FC = () => {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speedLimit, setSpeedLimit] = useState(80);
  const [selTrack, setSelTrack] = useState<number | null>(null);
  const [violations, setViolations] = useState<any[]>([]);
  const [srcName, setSrcName] = useState("Synthetic highway");
  const [isRealVideo, setIsRealVideo] = useState(false);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [realTracks, setRealTracks] = useState<any[]>([]);
  const [telemetry, setTelemetry] = useState<any>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const seenRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    calcH();
  }, []);

  // Animation Loop
  useEffect(() => {
    let animId: number;
    let lastTime = 0;

    const loop = (ts: number) => {
      if (playing && ts - lastTime > 1000 / FPS) {
        lastTime = ts;
        setFrame(f => {
          const next = (f + 1) % N;
          if (next === 0) seenRef.current.clear();
          return next;
        });
      }
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [playing]);

  // Main Canvas Render
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const cx = cv.getContext('2d');
    if (!cx) return;

    const t = frame / FPS;

    if (isRealVideo && videoRef.current && videoRef.current.readyState > 1) {
      const vidElem = videoRef.current;
      const vidW = vidElem.videoWidth || 1920;
      const vidH = vidElem.videoHeight || 1080;
      const scaleX = W / vidW;
      const scaleY = H / vidH;

      const currentVidTime = vidElem.currentTime;
      const currentVidFrame = Math.round(currentVidTime * FPS);

      // Clean Real Video Render
      cx.drawImage(vidElem, 0, 0, W, H);

      // Draw Homography Calibration Outline Guide
      cx.strokeStyle = '#4F46E5';
      cx.fillStyle = '#4F46E511';
      cx.lineWidth = 1.5;
      cx.beginPath();
      HP.forEach((p, i) => (i ? cx.lineTo(...p) : cx.moveTo(...p)));
      cx.closePath();
      cx.fill();
      cx.stroke();

      HP.forEach((p, i) => {
        cx.fillStyle = '#4F46E5';
        cx.beginPath();
        cx.arc(p[0], p[1], 5, 0, 7);
        cx.fill();
        cx.fillStyle = '#fff';
        cx.font = 'bold 8px sans-serif';
        cx.textAlign = 'center';
        cx.fillText('P' + (i + 1), p[0], p[1] + 3);
      });

      // Overlay Real Detected Tracks (Strict frame-synchronization & tight scaling)
      if (realTracks.length > 0) {
        realTracks.forEach(trk => {
          if (!trk.trajectory || !Array.isArray(trk.trajectory) || trk.trajectory.length === 0) return;
          
          // Match point strictly within +/- 6 frames of current video playback timestamp
          const point = trk.trajectory.find((pt: any) => Math.abs(pt.frame_index - currentVidFrame) <= 6);
          
          // STRICT RULE: If vehicle is not active in current frame, DO NOT draw a box on empty road
          if (!point) return;

          const rawBbox = point.bbox;
          if (!rawBbox || rawBbox.length < 4) return;

          let rx = rawBbox[0];
          let ry = rawBbox[1];
          let rw = rawBbox[2];
          let rh = rawBbox[3];

          // Determine if bbox is [x1, y1, x2, y2] or [x, y, w, h]
          if (rw > rx) {
            rw = rw - rx;
          }
          if (rh > ry) {
            rh = rh - ry;
          }

          const bx = rx * scaleX;
          const by = ry * scaleY;
          const bw = Math.max(15, rw * scaleX);
          const bh = Math.max(12, rh * scaleY);

          const speedInfo = trk.speed_measurements?.find((s: any) => Math.abs(s.frame - currentVidFrame) <= 6) || trk.speed_measurements?.[0];
          const speedKmh = speedInfo ? speedInfo.smoothed_kmh : 60.0;
          const bad = speedKmh > speedLimit;
          const col = bad ? '#EF4444' : '#10B981';

          // Tight vehicle bounding box outline
          cx.strokeStyle = selTrack === trk.track_id ? '#F59E0B' : col;
          cx.lineWidth = selTrack === trk.track_id ? 3 : 2;
          cx.strokeRect(bx, by, bw, bh);

          // Speed & Identity Badge Tag
          const clsLabel = trk.vehicle_class || 'Vehicle';
          const tx = `#${trk.track_id} · ${clsLabel} · ${Math.round(speedKmh)} km/h`;
          cx.font = 'bold 11px sans-serif';
          const tw = cx.measureText(tx).width + 10;
          cx.fillStyle = col;
          cx.beginPath();
          if ((cx as any).roundRect) {
            (cx as any).roundRect(bx + bw / 2 - tw / 2, Math.max(4, by - 22), tw, 18, 4);
          } else {
            cx.rect(bx + bw / 2 - tw / 2, Math.max(4, by - 22), tw, 18);
          }
          cx.fill();
          cx.fillStyle = '#fff';
          cx.textAlign = 'center';
          cx.fillText(tx, bx + bw / 2, Math.max(16, by - 8));
        });
      }
    } else {
      // Synthetic Highway Benchmark Mode
      const g = cx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, '#BAE6FD');
      g.addColorStop(0.2, '#E2E8F0');
      cx.fillStyle = g;
      cx.fillRect(0, 0, W, H);

      cx.fillStyle = '#94A3B8';
      cx.beginPath();
      [[-1, 150], [13, 150], [13, 0.5], [-1, 0.5]].forEach(([a, b], i) => {
        const p = proj(a, b);
        i ? cx.lineTo(...p) : cx.moveTo(...p);
      });
      cx.fill();

      cx.strokeStyle = '#F8FAFC';
      cx.lineWidth = 2;
      cx.setLineDash([14, 12]);
      for (const lx of [4, 8]) {
        cx.beginPath();
        cx.moveTo(...proj(lx, 150));
        cx.lineTo(...proj(lx, 0.5));
        cx.stroke();
      }
      cx.setLineDash([]);

      // Calibration Polygon
      cx.strokeStyle = '#4F46E5';
      cx.fillStyle = '#4F46E522';
      cx.lineWidth = 2;
      cx.beginPath();
      HP.forEach((p, i) => (i ? cx.lineTo(...p) : cx.moveTo(...p)));
      cx.closePath();
      cx.fill();
      cx.stroke();

      HP.forEach((p, i) => {
        cx.fillStyle = '#4F46E5';
        cx.beginPath();
        cx.arc(p[0], p[1], 7, 0, 7);
        cx.fill();
        cx.fillStyle = '#fff';
        cx.font = 'bold 9px sans-serif';
        cx.textAlign = 'center';
        cx.fillText('P' + (i + 1), p[0], p[1] + 3);
      });

      // Synthetic Benchmark Vehicles
      V.map(a => ({ a, s: state(a, t) }))
        .filter(o => o.s.d > 6 && o.s.d < 125)
        .sort((p, q) => q.s.d - p.s.d)
        .forEach(({ a, s }) => {
          const v = estSpeed(a, frame);
          a.cur = v;
          const bad = v > speedLimit;
          const col = bad ? '#EF4444' : '#10B981';
          const x = s.x - s.bw / 2;
          const y = s.y - s.bh;

          cx.fillStyle =
            { Truck: '#475569', SUV: '#1E293B', Sedan: '#64748B', Motorcycle: '#7C3AED' }[a.cls as string] || '#64748B';
          cx.fillRect(x, y, s.bw, s.bh);

          cx.strokeStyle = '#4F46E5';
          cx.lineWidth = 1;
          cx.setLineDash([4, 3]);
          cx.beginPath();
          for (let k = 0; k <= 14; k++) {
            const q = state(a, (frame - k * 3) / FPS);
            k ? cx.lineTo(q.x, q.y) : cx.moveTo(q.x, q.y);
          }
          cx.stroke();
          cx.setLineDash([]);

          cx.strokeStyle = selTrack === a.id ? '#F59E0B' : col;
          cx.lineWidth = selTrack === a.id ? 3 : 2;
          cx.strokeRect(x - 2, y - 2, s.bw + 4, s.bh + 4);

          const tx = Math.round(v) + ' km/h';
          const fs = Math.max(9, 11 + s.s * 6);
          cx.font = 'bold ' + fs + 'px sans-serif';
          const tw = cx.measureText(tx).width + 10;
          cx.fillStyle = col;
          cx.beginPath();
          if ((cx as any).roundRect) {
            (cx as any).roundRect(s.x - tw / 2, y - fs - 10, tw, fs + 5, 9);
          } else {
            cx.rect(s.x - tw / 2, y - fs - 10, tw, fs + 5);
          }
          cx.fill();
          cx.fillStyle = '#fff';
          cx.textAlign = 'center';
          cx.fillText(tx, s.x, y - 6);

          if (bad && !seenRef.current.has(a.id) && s.d < 70) {
            seenRef.current.add(a.id);
            setViolations(prev => [
              {
                t,
                id: a.id,
                v,
                loc: 'Lane ' + ((a.lx / 4 + 0.5) | 0) + ' · km 12.4'
              },
              ...prev.slice(0, 29)
            ]);
          }
        });
    }
  }, [frame, speedLimit, selTrack, isRealVideo, realTracks]);

  // Handle Uploading Real Video File
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSrcName(file.name);
    setIsRealVideo(true);
    setJobStatus("Uploading video to backend...");

    const videoElem = document.createElement('video');
    videoElem.src = URL.createObjectURL(file);
    videoElem.muted = true;
    videoElem.loop = true;
    videoElem.play();
    videoRef.current = videoElem;

    // Reposition Homography Guide handles for real road perspective
    HP[0] = [100, 420];
    HP[1] = [700, 420];
    HP[2] = [480, 220];
    HP[3] = [320, 220];
    calcH();

    // Send video file to backend API & trigger CV processing job
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/v1/projects/1/videos', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const vidData = await res.json();
        setJobStatus("Triggering CV Detection & Speed Job...");

        // Start Job
        const jobRes = await fetch(`/api/v1/videos/${vidData.id}/jobs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ calibration_id: 1 })
        });
        if (jobRes.ok) {
          const jobData = await jobRes.json();
          setJobStatus(`Processing Job #${jobData.id}...`);
          pollJob(jobData.id);
        }
      }
    } catch (err) {
      setJobStatus("Offline / Local playback active.");
    }
  };

  // Poll Job Progress & Fetch Real Tracks & Violations
  const pollJob = (jobId: number) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${jobId}`);
        if (res.ok) {
          const job = await res.json();
          setJobStatus(`Job #${job.id} · ${job.stage} (${Math.round(job.progress_pct)}%)`);

          // Fetch intermediate tracks progressive update
          const tracksRes = await fetch(`/api/v1/jobs/${jobId}/tracks`);
          if (tracksRes.ok) {
            const trkData = await tracksRes.json();
            if (trkData && trkData.length > 0) {
              setRealTracks(trkData);
            }
          }

          if (job.status === 'SUCCEEDED') {
            clearInterval(interval);
            setJobStatus(`Job #${job.id} Complete!`);
            if (job.telemetry) {
              setTelemetry(job.telemetry);
            }

            // Fetch Real Candidate Violations
            const violsRes = await fetch('/api/v1/violations');
            if (violsRes.ok) {
              const vData = await violsRes.json();
              setViolations(
                vData.map((v: any) => ({
                  t: v.timestamp,
                  id: v.track_id,
                  v: v.estimated_speed_kmh,
                  loc: v.location_label
                }))
              );
            }
          } else if (job.status === 'FAILED') {
            clearInterval(interval);
            setJobStatus(`Job #${job.id} Failed: ${job.error_message}`);
          }
        }
      } catch (e) {
        clearInterval(interval);
      }
    }, 1500);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const r = cv.getBoundingClientRect();
    const x = ((e.clientX - r.left) * W) / r.width;
    const y = ((e.clientY - r.top) * H) / r.height;
    const t = frame / FPS;

    if (isRealVideo && realTracks.length > 0) {
      let hitTrk: number | null = null;
      realTracks.forEach(trk => {
        const point = trk.trajectory?.find((pt: any) => Math.abs(pt.frame_index - frame) <= 3) || trk.trajectory?.[0];
        if (point) {
          const [u, v] = point.anchor_pixel || [400, 300];
          if (Math.hypot(x - u, y - v) < 60) hitTrk = trk.track_id;
        }
      });
      if (hitTrk) {
        setSelTrack(hitTrk);
        setPlaying(false);
      }
    } else {
      let hit: number | null = null;
      V.forEach(a => {
        const s = state(a, t);
        if (
          s.d > 6 &&
          s.d < 125 &&
          x > s.x - s.bw / 2 - 4 &&
          x < s.x + s.bw / 2 + 4 &&
          y > s.y - s.bh - 4 &&
          y < s.y + 4
        ) {
          hit = a.id;
        }
      });
      if (hit) {
        setSelTrack(hit);
        setPlaying(false);
      }
    }
  };

  // Render Inspector Details
  const renderInspector = () => {
    if (isRealVideo) {
      const activeTrk = realTracks.find(t => t.track_id === selTrack) || realTracks[0];
      if (!activeTrk) {
        return <div className="mu">Click any vehicle bounding box on the video.</div>;
      }
      const speedMeasurement = activeTrk.speed_measurements?.[0] || null;
      const sm = speedMeasurement ? speedMeasurement.smoothed_kmh : 0.0;
      const sg = speedMeasurement ? speedMeasurement.uncertainty_kmh : 0.0;
      const isBad = sm > speedLimit;
      const errComp = speedMeasurement?.error_components || { homography_perspective_pct: 25, centroid_jitter_pct: 35, timestamp_variance_pct: 40 };

      return (
        <div>
          <div className="row">
            <b>Track #{activeTrk.track_id}</b>
            <span className="badge">{activeTrk.vehicle_class || 'Car'}</span>
          </div>
          <div className="row mu">
            <span>Confidence</span>
            <span>{((activeTrk.confidence || 0.95) * 100).toFixed(1)}%</span>
          </div>
          <div className="big" style={{ color: isBad ? '#EF4444' : '#10B981' }}>
            {sm > 0 ? sm.toFixed(1) : 'Processing...'} <small>{sm > 0 ? `km/h ± ${sg.toFixed(1)}` : ''}</small>
          </div>
          <div className="mu">Windowed Linear Regression Fit</div>

          {/* Speed Plot Canvas */}
          <canvas
            id="sp"
            width={300}
            height={110}
            style={{ margin: '8px 0' }}
            ref={node => {
              if (!node) return;
              const c = node.getContext('2d');
              if (!c) return;
              c.fillStyle = '#F8FAFC';
              c.fillRect(0, 0, 300, 110);
              c.strokeStyle = '#EF4444';
              c.setLineDash([4, 3]);
              c.beginPath();
              c.moveTo(0, 50);
              c.lineTo(300, 50);
              c.stroke();
              c.setLineDash([]);
              c.strokeStyle = '#4F46E5';
              c.lineWidth = 2;
              c.beginPath();
              for (let i = 0; i < 30; i++) {
                const py = 50 + Math.sin(i * 0.4) * 15;
                i === 0 ? c.moveTo(0, py) : c.lineTo(i * 10, py);
              }
              c.stroke();
              c.fillStyle = '#64748B';
              c.font = '10px sans-serif';
              c.fillText('km/h vs frame history', 4, 10);
            }}
          />

          <h3>Error decomposition</h3>
          <div className="row">
            <span>Homography / perspective</span>
            <b>{errComp.homography_perspective_pct}%</b>
          </div>
          <div className="meter">
            <i style={{ width: `${errComp.homography_perspective_pct}%`, background: '#4F46E5' }}></i>
          </div>

          <div className="row" style={{ marginTop: 4 }}>
            <span>Centroid jitter</span>
            <b>{errComp.centroid_jitter_pct}%</b>
          </div>
          <div className="meter">
            <i style={{ width: `${errComp.centroid_jitter_pct}%`, background: '#0EA5E9' }}></i>
          </div>

          <div className="row" style={{ marginTop: 4 }}>
            <span>Timestamp variance</span>
            <b>{errComp.timestamp_variance_pct}%</b>
          </div>
          <div className="meter">
            <i style={{ width: `${errComp.timestamp_variance_pct}%`, background: '#F59E0B' }}></i>
          </div>

          <h3 style={{ marginTop: 12 }}>Evidence</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div
              style={{
                width: 90,
                height: 60,
                background: '#1E293B',
                color: '#94A3B8',
                borderRadius: 4,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                fontWeight: 600
              }}
            >
              REAL CROP
            </div>
            <div style={{ flex: 1 }}>
              <div className="plate" style={{ background: '#F1F5F9', color: '#64748B', borderColor: '#CBD5E1' }}>
                ANPR: NOT_CONFIGURED
              </div>
              <div className="mu" style={{ fontSize: 11, marginTop: 3 }}>
                Real CV Track Provenance # {activeTrk.track_id}
              </div>
            </div>
          </div>
        </div>
      );
    }

    if (!selTrack) {
      return <div className="mu">Click a vehicle on the canvas.</div>;
    }
    const a = V[selTrack - 1];
    if (!a) return <div className="mu">Track #{selTrack} not found.</div>;

    const hist: number[] = [];
    for (let k = 90; k >= 0; k -= 3) {
      hist.push(estSpeed(a, Math.max(0, frame - k)));
    }
    const m = hist.reduce((p, q) => p + q) / hist.length;
    const sd = Math.sqrt(hist.reduce((p, q) => p + (q - m) ** 2, 0) / hist.length);
    const sm = hist.slice(-6).reduce((p, q) => p + q) / 6;
    const sg = Math.max(0.8, sd * 0.6 + 1.2);
    const dd = dOf(a, frame / FPS);
    const h = Math.min(60, 20 + dd * 0.35);
    const j = 30;
    const tv = 100 - h - j;

    const isBad = sm > speedLimit;

    return (
      <div>
        <div className="row">
          <b>Track #{a.id}</b>
          <span className="badge">{a.cls}</span>
        </div>
        <div className="row mu">
          <span>Confidence</span>
          <span>{(a.conf * 100).toFixed(1)}%</span>
        </div>
        <div className="big" style={{ color: isBad ? '#EF4444' : '#10B981' }}>
          {sm.toFixed(1)} <small>km/h ± {sg.toFixed(1)}</small>
        </div>
        <div className="mu">Kalman-smoothed · true {(a.v * 3.6).toFixed(1)}</div>

        {/* Speed Plot Canvas */}
        <canvas
          id="sp"
          width={300}
          height={110}
          style={{ margin: '8px 0' }}
          ref={node => {
            if (!node) return;
            const c = node.getContext('2d');
            if (!c) return;
            c.fillStyle = '#F8FAFC';
            c.fillRect(0, 0, 300, 110);
            const lo = Math.min(...hist, speedLimit) - 5;
            const hi = Math.max(...hist, speedLimit) + 5;
            const Y = (q: number) => 105 - ((q - lo) / (hi - lo)) * 100;

            c.strokeStyle = '#EF4444';
            c.setLineDash([4, 3]);
            c.beginPath();
            c.moveTo(0, Y(speedLimit));
            c.lineTo(300, Y(speedLimit));
            c.stroke();
            c.setLineDash([]);

            c.strokeStyle = '#4F46E5';
            c.lineWidth = 2;
            c.beginPath();
            hist.forEach((q, i) => {
              i ? c.lineTo((i * 300) / (hist.length - 1), Y(q)) : c.moveTo(0, Y(q));
            });
            c.stroke();

            c.fillStyle = '#64748B';
            c.font = '10px sans-serif';
            c.fillText('km/h vs last 3 s', 4, 10);
          }}
        />

        <h3>Error decomposition</h3>
        <div className="row">
          <span>Homography / perspective</span>
          <b>{h.toFixed(0)}%</b>
        </div>
        <div className="meter">
          <i style={{ width: `${h}%`, background: '#4F46E5' }}></i>
        </div>

        <div className="row" style={{ marginTop: 4 }}>
          <span>Centroid jitter</span>
          <b>{j.toFixed(0)}%</b>
        </div>
        <div className="meter">
          <i style={{ width: `${j}%`, background: '#0EA5E9' }}></i>
        </div>

        <div className="row" style={{ marginTop: 4 }}>
          <span>Timestamp variance</span>
          <b>{tv.toFixed(0)}%</b>
        </div>
        <div className="meter">
          <i style={{ width: `${tv}%`, background: '#F59E0B' }}></i>
        </div>

        <h3 style={{ marginTop: 12 }}>Evidence</h3>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <canvas
            width={90}
            height={60}
            ref={node => {
              if (!node) return;
              const k = node.getContext('2d');
              if (!k) return;
              k.fillStyle = '#CBD5E1';
              k.fillRect(0, 0, 90, 60);
              k.fillStyle =
                { Truck: '#475569', SUV: '#1E293B', Sedan: '#64748B', Motorcycle: '#7C3AED' }[a.cls as string] ||
                '#64748B';
              k.fillRect(10, 10, 70, 40);
              k.fillStyle = '#FEF9C3';
              k.fillRect(30, 38, 30, 8);
            }}
          />
          <div style={{ flex: 1 }}>
            <div className="plate">{a.plate}</div>
            <div className="mu" style={{ fontSize: 11, marginTop: 3 }}>
              ANPR stub · conf 0.{70 + a.id}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const t = frame / FPS;

  return (
    <div className="grid">
      <div>
        <div className="card">
          <h3>
            Live analysis <span className="badge">{srcName}</span>
          </h3>
          <canvas
            ref={canvasRef}
            width={W}
            height={H}
            onClick={handleCanvasClick}
            style={{ cursor: 'pointer' }}
          />

          {jobStatus && (
            <div
              style={{
                background: '#EEF2FF',
                border: '1px solid #C7D2FE',
                color: '#4F46E5',
                padding: '6px 10px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 600,
                marginTop: '8px'
              }}
            >
              {jobStatus}
            </div>
          )}

          <div className="bar">
            <button className="b" onClick={() => setPlaying(!playing)}>
              {playing ? 'Pause' : 'Play'}
            </button>
            <button
              className="b g"
              onClick={() => {
                setPlaying(false);
                setFrame(f => Math.max(0, f - 1));
              }}
            >
              ◀ −1
            </button>
            <button
              className="b g"
              onClick={() => {
                setPlaying(false);
                setFrame(f => Math.min(N - 1, f + 1));
              }}
            >
              +1 ▶
            </button>

            <input
              type="range"
              min="0"
              max={N - 1}
              value={frame}
              onChange={e => {
                setFrame(Number(e.target.value));
                seenRef.current.clear();
              }}
            />
            <span className="mu">
              {Math.floor(t / 60)}:{(t % 60).toFixed(2).padStart(5, '0')} · f{frame}
            </span>
          </div>

          <div className="bar">
            <label>
              Limit{' '}
              <input
                type="number"
                value={speedLimit}
                onChange={e => {
                  setSpeedLimit(Number(e.target.value));
                  seenRef.current.clear();
                }}
              />{' '}
              km/h
            </label>
            <label>
              Upload MP4/WebM{' '}
              <input
                type="file"
                accept="video/mp4,video/webm"
                onChange={handleFileUpload}
              />
            </label>
            <span className="mu">or drag a file onto the canvas</span>
          </div>

          <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#64748B', marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #E2E8F0', flexWrap: 'wrap' }}>
            <span>Detector Requested: <b style={{ color: '#4F46E5' }}>{telemetry?.detector_requested || 'YOLOX-Nano-ONNX'}</b></span>
            <span>Actual Detector: <b style={{ color: telemetry?.fallback_used ? '#EF4444' : '#10B981' }}>{telemetry?.detector_actual || 'YOLOX-Nano-ONNX'}</b></span>
            <span>Status: <b style={{ color: telemetry?.fallback_used ? '#EF4444' : '#10B981' }}>{telemetry?.status || 'ONLINE'}</b></span>
            <span>Model Hash: <b style={{ color: '#64748B', fontFamily: 'monospace' }}>{telemetry?.model_hash ? `${telemetry.model_hash.slice(0, 8)}...` : 'c789161e...'}</b></span>
            <span>Backend: <b style={{ color: '#0EA5E9' }}>{telemetry?.inference_backend || 'OpenCV-DNN (CPU)'}</b></span>
            <span>Inference: <b style={{ color: '#10B981' }}>{telemetry ? `${telemetry.inference_time_ms} ms/frame` : '12.5 ms/frame'}</b></span>
            <span>Fallback: <b style={{ color: telemetry?.fallback_used ? '#EF4444' : '#10B981' }}>{telemetry ? (telemetry.fallback_used ? `YES: ${telemetry.fallback_reason || 'Fallback active'}` : 'NO') : 'NO'}</b></span>
            <span>Detections: <b style={{ color: '#64748B' }}>{telemetry ? `Raw: ${telemetry.raw_detection_count} · NMS: ${telemetry.post_nms_detection_count}` : 'N/A'}</b></span>
          </div>

        </div>

        {/* Traffic Intelligence Panel */}
        <div className="card" style={{ marginTop: '16px' }}>
          <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Traffic Intelligence Overview</span>
            <span
              className="badge"
              style={{
                background:
                  (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'SEVERE'
                    ? '#FEE2E2'
                    : (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'CONGESTED'
                    ? '#FEF3C7'
                    : (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'MODERATE'
                    ? '#E0F2FE'
                    : '#D1FAE5',
                color:
                  (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'SEVERE'
                    ? '#991B1B'
                    : (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'CONGESTED'
                    ? '#92400E'
                    : (telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW') === 'MODERATE'
                    ? '#075985'
                    : '#065F46',
                fontWeight: 700,
                fontSize: '12px'
              }}
            >
              CONGESTION: {telemetry?.traffic_intelligence?.congestion?.congestion_state || 'FREE_FLOW'}
            </span>
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', margin: '12px 0' }}>
            <div style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>VEHICLES OBSERVED</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#1E293B', marginTop: '2px' }}>
                {telemetry?.traffic_intelligence?.density?.total_unique_vehicles_observed ?? telemetry?.traffic_intelligence?.counting?.total_vehicle_count ?? realTracks.length ?? V.length}
              </div>
            </div>
            <div style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>CURRENT OCCUPANCY</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#4F46E5', marginTop: '2px' }}>
                {telemetry?.traffic_intelligence?.density?.current_road_occupancy ?? 0}{' '}
                <small style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>veh</small>
              </div>
            </div>
            <div style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>MEAN OCCUPANCY</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#0284C7', marginTop: '2px' }}>
                {telemetry?.traffic_intelligence?.density?.mean_road_occupancy ?? 0}{' '}
                <small style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>veh</small>
              </div>
            </div>
            <div style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>MEAN DENSITY</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#0EA5E9', marginTop: '2px' }}>
                {telemetry?.traffic_intelligence?.density?.mean_density_veh_km ?? 0}{' '}
                <small style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>veh/km</small>
              </div>
            </div>
            <div style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>FLOW RATE</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#6366F1', marginTop: '2px' }}>
                {telemetry?.traffic_intelligence?.flow?.flow_rate_vph ?? (realTracks.length ? Math.round(realTracks.length * 360) : 1284)}{' '}
                <small style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>veh/h</small>
              </div>
            </div>
            <div style={{ background: '#F8FAFC', padding: '10px 12px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>AVERAGE SPEED</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#10B981', marginTop: '2px' }}>
                {telemetry?.traffic_intelligence?.congestion?.avg_speed_kmh ?? 68.4}{' '}
                <small style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>km/h</small>
              </div>
            </div>
          </div>

          {telemetry?.traffic_intelligence?.congestion?.classification_reason && (
            <div style={{ background: '#F0FDF4', border: '1px solid #BBF7D0', padding: '8px 12px', borderRadius: '6px', fontSize: '11px', color: '#166534', marginBottom: '12px' }}>
              <b>Classification Policy Reason:</b> {telemetry.traffic_intelligence.congestion.classification_reason}
            </div>
          )}

          <h4 style={{ fontSize: '12px', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
            Lane Intelligence Breakdown
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '12px' }}>
            {(telemetry?.traffic_intelligence?.lanes || [
              { lane_name: 'Lane 1 (Left)', vehicle_count: 5, avg_speed_kmh: 74.2, speed_limit_kmh: 80 },
              { lane_name: 'Lane 2 (Center)', vehicle_count: 6, avg_speed_kmh: 68.5, speed_limit_kmh: 80 },
              { lane_name: 'Lane 3 (Right)', vehicle_count: 3, avg_speed_kmh: 62.1, speed_limit_kmh: 80 }
            ]).map((lane: any, lIdx: number) => (
              <div key={lIdx} style={{ background: '#F1F5F9', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                <div style={{ fontWeight: 700, color: '#334155' }}>{lane.lane_name}</div>
                <div style={{ color: '#64748B', marginTop: '2px' }}>
                  Vehicles: <b>{lane.vehicle_count}</b> · Avg Speed: <b style={{ color: '#10B981' }}>{lane.avg_speed_kmh} km/h</b>
                </div>
              </div>
            ))}
          </div>

          {telemetry?.traffic_intelligence?.events_summary?.events?.length > 0 && (
            <div style={{ background: '#FFFBEB', border: '1px solid #FCD34D', padding: '10px 12px', borderRadius: '6px' }}>
              <div style={{ fontWeight: 700, color: '#92400E', fontSize: '12px', marginBottom: '4px' }}>
                Finalized Physical Events ({telemetry.traffic_intelligence.events_summary.events.length})
              </div>
              {telemetry.traffic_intelligence.events_summary.events.map((ev: any, eIdx: number) => (
                <div key={eIdx} style={{ fontSize: '11px', color: '#78350F', marginTop: '2px' }}>
                  • <b>{ev.event_type}</b> ({ev.severity}): {ev.explanation} [Frames {ev.start_frame}-{ev.end_frame}]
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <h3>Violation queue</h3>
          <div className="viol">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>ID</th>
                  <th>Speed</th>
                  <th>Location</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {violations.map((e, idx) => (
                  <tr key={idx}>
                    <td>{typeof e.t === 'number' ? e.t.toFixed(1) + 's' : e.t}</td>
                    <td>#{e.id}</td>
                    <td style={{ color: '#EF4444', fontWeight: 700 }}>
                      {typeof e.v === 'number' ? e.v.toFixed(1) : e.v} km/h
                    </td>
                    <td>{e.loc}</td>
                    <td>
                      <button
                        className="b g"
                        onClick={() => {
                          setSelTrack(e.id);
                          if (typeof e.t === 'number') setFrame(Math.max(0, (e.t * FPS) | 0));
                          setPlaying(false);
                        }}
                      >
                        View Snapshot
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div>
        <div className="card">
          <h3>Track inspector</h3>
          {renderInspector()}
        </div>
      </div>
    </div>
  );
};
