import React, { useState, useEffect, useRef } from 'react';

const W = 800;
const H = 450;

const S = (d: number) => 1 / (1 + d / 25);
const proj = (lx: number, d: number): [number, number] => [400 + (lx - 6) * S(d) * 60, 110 + 330 * S(d)];

const CP = [[0, 10], [12, 10], [12, 90], [0, 90]];
const HP_def = [[0, 10], [12, 10], [12, 90], [0, 90]].map(([a, b]) => proj(a, b));

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

export const CalibrationView: React.FC = () => {
  const [HP, setHP] = useState<[number, number][]>(() => HP_def.map(p => [...p] as [number, number]));
  const [camHeight, setCamHeight] = useState(9);
  const [camPitch, setCamPitch] = useState(14);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragIdxRef = useRef<number>(-1);

  // Compute Homography matrix
  const calcH = () => {
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
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  };

  const Hm = calcH();

  const toW = (x: number, y: number) => {
    const q = Hm[2][0] * x + Hm[2][1] * y + 1;
    return [(Hm[0][0] * x + Hm[0][1] * y + Hm[0][2]) / q, (Hm[1][0] * x + Hm[1][1] * y + Hm[1][2]) / q];
  };

  // Draw Calibration Canvas
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const c2 = cv.getContext('2d');
    if (!c2) return;

    // Draw Road
    const g = c2.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#BAE6FD');
    g.addColorStop(0.2, '#E2E8F0');
    c2.fillStyle = g;
    c2.fillRect(0, 0, W, H);

    c2.fillStyle = '#94A3B8';
    c2.beginPath();
    [[-1, 150], [13, 150], [13, 0.5], [-1, 0.5]].forEach(([a, b], i) => {
      const p = proj(a, b);
      i ? c2.lineTo(...p) : c2.moveTo(...p);
    });
    c2.fill();

    c2.strokeStyle = '#F8FAFC';
    c2.lineWidth = 2;
    c2.setLineDash([14, 12]);
    for (const lx of [4, 8]) {
      c2.beginPath();
      c2.moveTo(...proj(lx, 150));
      c2.lineTo(...proj(lx, 0.5));
      c2.stroke();
    }
    c2.setLineDash([]);

    // Draw Polygon
    c2.strokeStyle = '#4F46E5';
    c2.fillStyle = '#4F46E522';
    c2.lineWidth = 2;
    c2.beginPath();
    HP.forEach((p, i) => (i ? c2.lineTo(...p) : c2.moveTo(...p)));
    c2.closePath();
    c2.fill();
    c2.stroke();

    HP.forEach((p, i) => {
      c2.fillStyle = '#4F46E5';
      c2.beginPath();
      c2.arc(p[0], p[1], 7, 0, 7);
      c2.fill();
      c2.fillStyle = '#fff';
      c2.font = 'bold 9px sans-serif';
      c2.textAlign = 'center';
      c2.fillText('P' + (i + 1), p[0], p[1] + 3);
    });

    // Draw Reference Markers
    const mk = [[2, 20], [10, 30], [6, 45], [2, 60], [10, 75], [6, 85]];
    c2.fillStyle = '#F59E0B';
    mk.forEach(([lx, d]) => {
      const p = proj(lx, d);
      c2.beginPath();
      c2.arc(p[0], p[1], 4, 0, 7);
      c2.fill();
    });
  }, [HP]);

  // Handle Dragging
  const pos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cv = canvasRef.current;
    if (!cv) return [0, 0];
    const r = cv.getBoundingClientRect();
    return [((e.clientX - r.left) * W) / r.width, ((e.clientY - r.top) * H) / r.height];
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const [x, y] = pos(e);
    const d = HP.findIndex(p => Math.hypot(p[0] - x, p[1] - y) < 14);
    if (d >= 0) {
      dragIdxRef.current = d;
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragIdxRef.current < 0) return;
    const [x, y] = pos(e);
    setHP(prev => {
      const next = prev.map(p => [...p] as [number, number]);
      next[dragIdxRef.current] = [x, y];
      return next;
    });
  };

  const handlePointerUp = () => {
    dragIdxRef.current = -1;
  };

  const handleReset = () => {
    setHP(HP_def.map(p => [...p] as [number, number]));
  };

  // Compute Metrics
  const mk = [[2, 20], [10, 30], [6, 45], [2, 60], [10, 75], [6, 85]];
  let se = 0;
  mk.forEach(([lx, d]) => {
    const p = proj(lx, d);
    const w = toW(...p);
    se += (w[0] - lx) ** 2 + (w[1] - d) ** 2;
  });
  const rm = Math.sqrt(se / mk.length);
  const sx = 12 / Math.hypot(HP[1][0] - HP[0][0], HP[1][1] - HP[0][1]);
  const sy = 80 / Math.hypot(HP[3][0] - HP[0][0], HP[3][1] - HP[0][1]);

  const matHtml = Hm.map(r => r.map(v => v.toExponential(3).padStart(11)).join(' ')).join('\n');

  return (
    <div className="grid">
      <div className="card">
        <h3>Drag P1–P4 on the road</h3>
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          style={{ touchAction: 'none', cursor: 'crosshair' }}
        />
        <p className="mu">Ground-plane rectangle is 12 m × 80 m. Handles are shared with the Workbench.</p>
      </div>

      <div>
        <div className="card">
          <h3>Homography H</h3>
          <div className="mat" style={{ whiteSpace: 'pre' }}>
            {matHtml}
          </div>
        </div>

        <div className="card">
          <h3>Scale &amp; error</h3>
          <div>
            <div className="row">
              <span>
                S<sub>x</sub> (m/px, near edge)
              </span>
              <b>{sx.toFixed(4)}</b>
            </div>
            <div className="row">
              <span>
                S<sub>y</sub> (m/px, mean)
              </span>
              <b>{sy.toFixed(4)}</b>
            </div>
            <div className="row">
              <span>Reprojection RMSE</span>
              <b style={{ color: rm < 1 ? '#10B981' : rm < 4 ? '#F59E0B' : '#EF4444' }}>
                {rm.toFixed(3)} m
              </b>
            </div>
          </div>

          <div className="row" style={{ marginTop: 8 }}>
            <span>Camera height (m)</span>
            <input
              type="number"
              value={camHeight}
              onChange={e => setCamHeight(Number(e.target.value))}
            />
          </div>
          <div className="row">
            <span>Pitch (°)</span>
            <input
              type="number"
              value={camPitch}
              onChange={e => setCamPitch(Number(e.target.value))}
            />
          </div>
          <button className="b g" style={{ marginTop: 8, width: '100%' }} onClick={handleReset}>
            Reset handles
          </button>
        </div>
      </div>
    </div>
  );
};
