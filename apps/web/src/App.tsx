import React, { useState, useEffect } from 'react';
import { WorkbenchView } from './components/WorkbenchView';
import { CalibrationView } from './components/CalibrationView';
import { ExperimentsView } from './components/ExperimentsView';
import { ArchitectureView } from './components/ArchitectureView';
import { BenchmarkView } from './components/BenchmarkView';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'wb' | 'cal' | 'exp' | 'arch' | 'bm'>('wb');
  const [healthStatus, setHealthStatus] = useState<string>('Online');

  useEffect(() => {
    fetch('/api/v1/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'online') {
          setHealthStatus(`Online (${data.cv_engine})`);
        }
      })
      .catch(() => setHealthStatus('Local'));
  }, []);

  return (
    <div>
      <header>
        <b>◎ VisionRadar</b>
        <nav id="nav">
          <button
            className={activeTab === 'wb' ? 'on' : ''}
            onClick={() => setActiveTab('wb')}
          >
            Workbench
          </button>
          <button
            className={activeTab === 'cal' ? 'on' : ''}
            onClick={() => setActiveTab('cal')}
          >
            Calibration
          </button>
          <button
            className={activeTab === 'bm' ? 'on' : ''}
            onClick={() => setActiveTab('bm')}
          >
            Speed Benchmark
          </button>
          <button
            className={activeTab === 'exp' ? 'on' : ''}
            onClick={() => setActiveTab('exp')}
          >
            Experiments &amp; Ground-Truth
          </button>
          <button
            className={activeTab === 'arch' ? 'on' : ''}
            onClick={() => setActiveTab('arch')}
          >
            Architecture &amp; APIs
          </button>
        </nav>
        <div style={{ marginLeft: 'auto', fontSize: '12px', opacity: 0.85 }}>
          Backend: {healthStatus}
        </div>
      </header>

      <main>
        <section className={`tab ${activeTab === 'wb' ? 'on' : ''}`} id="wb">
          <WorkbenchView />
        </section>

        <section className={`tab ${activeTab === 'cal' ? 'on' : ''}`} id="cal">
          <CalibrationView />
        </section>

        <section className={`tab ${activeTab === 'bm' ? 'on' : ''}`} id="bm">
          <BenchmarkView />
        </section>

        <section className={`tab ${activeTab === 'exp' ? 'on' : ''}`} id="exp">
          <ExperimentsView />
        </section>

        <section className={`tab ${activeTab === 'arch' ? 'on' : ''}`} id="arch">
          <ArchitectureView />
        </section>
      </main>
    </div>
  );
};
