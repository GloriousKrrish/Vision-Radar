import React, { useState, useEffect } from 'react';
import { apiUrl } from './config/api';
import { WorkbenchView } from './components/WorkbenchView';
import { CalibrationView } from './components/CalibrationView';
import { ExperimentsView } from './components/ExperimentsView';
import { ArchitectureView } from './components/ArchitectureView';
import { BenchmarkView } from './components/BenchmarkView';
import { ResearchWorkbenchView } from './components/ResearchWorkbenchView';
import { SharedJobProvider, useSharedJob } from './context/SharedJobContext';
import { ExcelExportModal } from './components/ExcelExportModal';

const AppContent: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'wb' | 'cal' | 'exp' | 'arch' | 'bm' | 'rwb'>('wb');
  const [healthStatus, setHealthStatus] = useState<string>('Online');
  const { activeJobId, activeJob, jobList, selectJob, openExportModal } = useSharedJob();

  useEffect(() => {
    fetch(apiUrl('/api/v1/health'))
      .then(res => res.json())
      .then(data => {
        if (data.status === 'online') {
          const verifiedStr = data.pipeline_verified ? ` · Verified (${data.succeeded_jobs} jobs)` : '';
          setHealthStatus(`Online (${data.cv_engine}${verifiedStr})`);
        }
      })
      .catch(() => setHealthStatus('Local'));
  }, []);

  return (
    <div>
      <header>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <b>◎ VisionRadar</b>
          
          {/* Shared Active Job Context Selector (Phase B) */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(255, 255, 255, 0.15)',
              padding: '4px 10px',
              borderRadius: '8px',
              fontSize: '12px'
            }}
          >
            <span style={{ opacity: 0.85, fontWeight: 600 }}>Active Job:</span>
            <select
              value={activeJobId || ''}
              onChange={e => {
                const val = parseInt(e.target.value, 10);
                if (!isNaN(val)) selectJob(val);
              }}
              style={{
                background: 'rgba(15, 23, 42, 0.4)',
                color: '#ffffff',
                border: '1px solid rgba(255, 255, 255, 0.25)',
                borderRadius: '6px',
                padding: '2px 8px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              {jobList.length === 0 && <option value="">No jobs found</option>}
              {jobList.map(j => {
                const isExam = (j.config_json?.module || '').toLowerCase().includes('exam');
                return (
                  <option key={j.id} value={j.id} style={{ color: '#0F172A', background: '#ffffff' }}>
                    #{j.id} {isExam ? '🎓 Exam' : '🚗 Traffic'} · {j.status}
                  </option>
                );
              })}
            </select>
            {activeJob && (
              <span
                style={{
                  background: activeJob.status === 'SUCCEEDED' ? '#10B981' : activeJob.status === 'RUNNING' ? '#3B82F6' : '#F59E0B',
                  color: '#ffffff',
                  fontSize: '10px',
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: '4px'
                }}
              >
                {activeJob.status}
              </span>
            )}
          </div>
        </div>

        <nav id="nav">
          <button
            className={activeTab === 'wb' ? 'on' : ''}
            onClick={() => setActiveTab('wb')}
          >
            Workbench
          </button>
          <button
            className={activeTab === 'rwb' ? 'on' : ''}
            onClick={() => setActiveTab('rwb')}
          >
            🔬 Research Workbench
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

        {/* Global Excel Export & System Status */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => openExportModal()}
            style={{
              background: '#10B981',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              padding: '6px 14px',
              fontWeight: 700,
              fontSize: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            title="Export Active Job to Professional Excel Workbook (.xlsx)"
          >
            <span>📊</span>
            <span>Excel Export</span>
          </button>

          <div style={{ fontSize: '12px', opacity: 0.85 }}>
            Backend: {healthStatus}
          </div>
        </div>
      </header>

      <main>
        <section className={`tab ${activeTab === 'wb' ? 'on' : ''}`} id="wb">
          <WorkbenchView />
        </section>

        <section className={`tab ${activeTab === 'rwb' ? 'on' : ''}`} id="rwb">
          <ResearchWorkbenchView />
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

      {/* Global Excel Export Modal (Phase G) */}
      <ExcelExportModal />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <SharedJobProvider>
      <AppContent />
    </SharedJobProvider>
  );
};


