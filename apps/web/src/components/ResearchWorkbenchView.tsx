import React, { useState, useEffect } from 'react';

export const ResearchWorkbenchView: React.FC = () => {
  const [subTab, setSubTab] = useState<'list' | 'detail' | 'compare' | 'ablations' | 'failures' | 'explain' | 'report'>('list');
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>('EXP-2026-000001');
  const [expDetail, setExpDetail] = useState<any>(null);
  const [expA, setExpA] = useState<string>('EXP-2026-000001');
  const [expB, setExpB] = useState<string>('EXP-2026-000002');
  const [compData, setCompData] = useState<any>(null);
  const [explainData, setExplainData] = useState<any>(null);
  const [selectedAnomalyId, setSelectedAnomalyId] = useState<string>('anom_EXP-2026-000001_1');
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [filterDataset, setFilterDataset] = useState<string>('ALL');
  const [filterGtStatus, setFilterGtStatus] = useState<string>('ALL');

  useEffect(() => {
    fetchExperiments();
  }, []);

  useEffect(() => {
    if (selectedExpId) {
      fetchExperimentDetail(selectedExpId);
    }
  }, [selectedExpId]);

  const fetchExperiments = () => {
    fetch('/api/v1/experiments')
      .then(res => res.json())
      .then(data => {
        setExperiments(data);
        if (data.length >= 2) {
          setExpA(data[0].experiment_id);
          setExpB(data[1].experiment_id);
        }
      })
      .catch(err => console.error(err));
  };

  const fetchExperimentDetail = (id: string) => {
    fetch(`/api/v1/experiments/${id}`)
      .then(res => res.json())
      .then(data => {
        setExpDetail(data);
        if (data.anomalies && data.anomalies.length > 0) {
          const aid = data.anomalies[0].anomaly_id;
          setSelectedAnomalyId(aid);
          fetchExplainability(id, aid);
        }
      })
      .catch(err => console.error(err));
  };

  const fetchComparison = () => {
    fetch(`/api/v1/experiments/compare?exp_a_id=${expA}&exp_b_id=${expB}`)
      .then(res => res.json())
      .then(data => setCompData(data))
      .catch(err => console.error(err));
  };

  const fetchExplainability = (expId: string, anomId: string) => {
    fetch(`/api/v1/experiments/${expId}/explainability/${anomId}`)
      .then(res => res.json())
      .then(data => setExplainData(data))
      .catch(err => console.error(err));
  };

  const fetchReport = (id: string) => {
    fetch(`/api/v1/experiments/${id}/report`)
      .then(res => res.json())
      .then(data => setReportMarkdown(data.report_markdown))
      .catch(err => console.error(err));
  };

  const handleReproduce = (id: string) => {
    fetch(`/api/v1/experiments/${id}/reproduce`, { method: 'POST' })
      .then(res => res.json())
      .then(data => alert(`Reproduction Triggered: ${data.message}`))
      .catch(err => alert(`Reproduction Failed: ${err}`));
  };

  const filteredExperiments = experiments.filter(e => {
    if (filterDataset !== 'ALL' && !e.dataset_name.includes(filterDataset)) return false;
    if (filterGtStatus !== 'ALL' && e.ground_truth_status !== filterGtStatus) return false;
    return true;
  });

  return (
    <div>
      {/* Sub Navigation Bar */}
      <div className="card" style={{ marginBottom: '14px', background: '#FFFFFF' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          <b style={{ color: '#4F46E5', fontSize: '15px' }}>🔬 RESEARCH WORKBENCH</b>
          <span style={{ color: '#94A3B8' }}>|</span>
          <button className={`g ${subTab === 'list' ? 'b' : ''}`} onClick={() => setSubTab('list')}>
            📋 Experiments Registry
          </button>
          <button className={`g ${subTab === 'detail' ? 'b' : ''}`} onClick={() => setSubTab('detail')}>
            🔍 Experiment Detail
          </button>
          <button className={`g ${subTab === 'compare' ? 'b' : ''}`} onClick={() => { setSubTab('compare'); fetchComparison(); }}>
            ⚖️ Scientific Compare
          </button>

          <button className={`g ${subTab === 'explain' ? 'b' : ''}`} onClick={() => { setSubTab('explain'); if (selectedExpId && selectedAnomalyId) fetchExplainability(selectedExpId, selectedAnomalyId); }}>
            💡 Anomaly Explainability
          </button>
          <button className={`g ${subTab === 'report' ? 'b' : ''}`} onClick={() => { setSubTab('report'); if (selectedExpId) fetchReport(selectedExpId); }}>
            📄 Research Report
          </button>
        </div>
      </div>

      {/* PAGE 1: EXPERIMENTS LIST */}
      {subTab === 'list' && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3>PERPETUAL RESEARCH EXPERIMENT REGISTRY</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              <select value={filterDataset} onChange={e => setFilterDataset(e.target.value)} style={{ width: '150px' }}>
                <option value="ALL">All Datasets</option>

                <option value="VNTraffic">VNTraffic</option>
                <option value="AICC22">AICC22 Custom</option>
                <option value="Traffic1">Traffic1</option>
                <option value="Synthetic">Synthetic</option>
              </select>

              <select value={filterGtStatus} onChange={e => setFilterGtStatus(e.target.value)} style={{ width: '180px' }}>
                <option value="ALL">All GT Tiers</option>
                <option value="GT-VALIDATED">GT-VALIDATED</option>
                <option value="ENGINEERING-VALIDATED">ENGINEERING-VALIDATED</option>
                <option value="QUALITATIVE">QUALITATIVE</option>
                <option value="NOT-EVALUABLE">NOT-EVALUABLE</option>
              </select>
            </div>
          </div>

          <table>
            <thead>
              <tr>
                <th>Experiment ID</th>
                <th>Experiment Name</th>
                <th>Dataset / Video</th>
                <th>Model</th>
                <th>GT Status</th>
                <th>Result Verdict</th>
                <th>Runtime</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredExperiments.map(e => (
                <tr key={e.experiment_id} style={{ background: selectedExpId === e.experiment_id ? '#EEF2FF' : 'transparent' }}>
                  <td><b style={{ color: '#4F46E5' }}>{e.experiment_id}</b></td>
                  <td>{e.name}</td>
                  <td>{e.video_filename}</td>
                  <td><span className="badge" style={{ background: '#64748B' }}>{e.model_id}</span></td>
                  <td>
                    <span className="badge" style={{
                      background: e.ground_truth_status === 'GT-VALIDATED' ? '#10B981' :
                        e.ground_truth_status === 'ENGINEERING-VALIDATED' ? '#0EA5E9' : '#F59E0B'
                    }}>
                      {e.ground_truth_status}
                    </span>
                  </td>
                  <td><small style={{ fontWeight: 600 }}>{e.result_status}</small></td>
                  <td>{e.duration_sec}s</td>
                  <td>
                    <button className="g" style={{ padding: '3px 8px', fontSize: '12px' }} onClick={() => { setSelectedExpId(e.experiment_id); setSubTab('detail'); }}>
                      Inspect →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* PAGE 2: EXPERIMENT DETAIL */}
      {subTab === 'detail' && expDetail && (
        <div>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2>{expDetail.name} <span style={{ color: '#4F46E5' }}>({expDetail.experiment_id})</span></h2>
                <div style={{ color: '#64748B', fontSize: '13px' }}>
                  Dataset: {expDetail.dataset_name} | Video: {expDetail.video_filename} | Resolution: {expDetail.resolution} | FPS: {expDetail.fps}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button className="b" onClick={() => handleReproduce(expDetail.experiment_id)}>
                  🔄 Reproduce Experiment
                </button>
              </div>
            </div>
          </div>

          <div className="kpi" style={{ marginBottom: '14px' }}>
            <div>
              <div className="mu">GROUND TRUTH TIER</div>
              <div style={{ fontSize: '16px', fontWeight: 700, color: '#0EA5E9' }}>{expDetail.ground_truth_status}</div>
            </div>
            <div>
              <div className="mu">ANOMALY CANDIDATES</div>
              <div className="big" style={{ color: '#4F46E5' }}>{expDetail.anomalies?.length || 0}</div>
            </div>
            <div>
              <div className="mu">FAILURE CASES</div>
              <div className="big" style={{ color: '#EF4444' }}>{expDetail.failure_cases?.length || 0}</div>
            </div>
            <div>
              <div className="mu">CONFIG HASH</div>
              <div style={{ fontSize: '11px', fontFamily: 'monospace', color: '#64748B' }}>
                {expDetail.configuration_hash ? expDetail.configuration_hash.substring(0, 16) + '...' : 'N/A'}
              </div>
            </div>
          </div>

          <div className="grid">
            <div>
              {/* Configuration Snapshot */}
              <div className="card">
                <h3>IMMUTABLE CONFIGURATION SNAPSHOT</h3>
                <pre>{JSON.stringify(expDetail.configuration, null, 2)}</pre>
              </div>

              {/* Metrics with Provenance */}
              <div className="card">
                <h3>METRIC PROVENANCE MATRIX</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Metric Name</th>
                      <th>Value</th>
                      <th>Source Engine</th>
                      <th>GT Classification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(expDetail.metrics || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td><b>{k}</b></td>
                        <td>{String(v)}</td>
                        <td><small style={{ color: '#64748B' }}>BehavioralAnomalyEngine</small></td>
                        <td><span className="badge" style={{ background: '#0EA5E9' }}>ENGINEERING-VALIDATED</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              {/* Dataset Provenance */}
              <div className="card">
                <h3>DATASET PROVENANCE</h3>
                <div className="row"><span className="mu">Filename:</span> <b>{expDetail.video_filename}</b></div>
                <div className="row"><span className="mu">Video Hash:</span> <small style={{ fontFamily: 'monospace' }}>{expDetail.video_hash ? expDetail.video_hash.substring(0, 12) : 'N/A'}</small></div>
                <div className="row"><span className="mu">Duration:</span> <b>{expDetail.duration_sec}s</b></div>
                <div className="row"><span className="mu">Frame Count:</span> <b>{expDetail.frame_count}</b></div>
                <div className="row"><span className="mu">GT Type:</span> <b>ENGINEERING_RULES</b></div>
              </div>

              {/* Failure Cases */}
              <div className="card">
                <h3>LOGGED FAILURE CASES ({expDetail.failure_cases?.length || 0})</h3>
                {expDetail.failure_cases && expDetail.failure_cases.length > 0 ? (
                  expDetail.failure_cases.map((fc: any, i: number) => (
                    <div key={i} style={{ borderBottom: '1px solid #E2E8F0', padding: '6px 0' }}>
                      <span className="badge" style={{ background: '#EF4444' }}>{fc.category}</span>
                      <div style={{ fontSize: '12px', marginTop: '4px' }}>{fc.description}</div>
                    </div>
                  ))
                ) : (
                  <div style={{ color: '#64748B', fontSize: '13px' }}>No failure cases recorded.</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PAGE 3: SCIENTIFIC COMPARISON */}
      {subTab === 'compare' && (
        <div className="card">
          <h3>SCIENTIFIC EXPERIMENT COMPARISON (NO WINNERS / NO RANKINGS)</h3>
          <div className="bar" style={{ marginBottom: '14px' }}>
            <label className="mu">Experiment A:</label>
            <select value={expA} onChange={e => setExpA(e.target.value)} style={{ width: '200px' }}>
              {experiments.map(e => <option key={e.experiment_id} value={e.experiment_id}>{e.experiment_id} - {e.video_filename}</option>)}
            </select>

            <label className="mu">Experiment B:</label>
            <select value={expB} onChange={e => setExpB(e.target.value)} style={{ width: '200px' }}>
              {experiments.map(e => <option key={e.experiment_id} value={e.experiment_id}>{e.experiment_id} - {e.video_filename}</option>)}
            </select>

            <button className="b" onClick={fetchComparison}>Compare Side-by-Side</button>
          </div>

          {compData && (
            <div>
              <table>
                <thead>
                  <tr>
                    <th>Field Category</th>
                    <th>Experiment A ({compData.experiment_a_id})</th>
                    <th>Experiment B ({compData.experiment_b_id})</th>
                    <th>Scientific Status</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(compData.metadata_comparison || {}).map(([k, item]: [string, any]) => (
                    <tr key={k}>
                      <td><b>Metadata: {k}</b></td>
                      <td>{String(item.val_a)}</td>
                      <td>{String(item.val_b)}</td>
                      <td>
                        <span className="badge" style={{ background: item.status === 'UNCHANGED' ? '#64748B' : '#4F46E5' }}>
                          {item.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {Object.entries(compData.metrics_comparison || {}).map(([k, item]: [string, any]) => (
                    <tr key={k}>
                      <td><b>Metric: {k}</b></td>
                      <td>{String(item.val_a)}</td>
                      <td>{String(item.val_b)}</td>
                      <td>
                        <span className="badge" style={{ background: item.status === 'UNCHANGED' ? '#64748B' : '#0EA5E9' }}>
                          {item.status} {item.numeric_delta !== undefined ? `(Δ ${item.numeric_delta})` : ''}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* PAGE 4: ANOMALY EXPLAINABILITY */}
      {subTab === 'explain' && (
        <div>
          <div className="card">
            <h3>EVIDENCE-BACKED ANOMALY EXPLAINABILITY ENGINE</h3>
            <div className="bar">
              <label className="mu">Experiment:</label>
              <select value={selectedExpId} onChange={e => { setSelectedExpId(e.target.value); fetchExperimentDetail(e.target.value); }} style={{ width: '180px' }}>
                {experiments.map(e => <option key={e.experiment_id} value={e.experiment_id}>{e.experiment_id}</option>)}
              </select>

              <label className="mu">Anomaly Candidate:</label>
              <select value={selectedAnomalyId} onChange={e => { setSelectedAnomalyId(e.target.value); fetchExplainability(selectedExpId, e.target.value); }} style={{ width: '220px' }}>
                {expDetail?.anomalies?.map((a: any) => (
                  <option key={a.anomaly_id} value={a.anomaly_id}>{a.anomaly_id} ({a.anomaly_type})</option>
                ))}
              </select>
            </div>
          </div>

          {explainData && (
            <div className="grid">
              <div className="card">
                <div className="row"><span className="mu">WHAT (Type):</span> <b>{explainData.what}</b></div>
                <div className="row"><span className="mu">WHEN (Timestamp):</span> <b>{explainData.when_timestamp_sec}s (Frames {explainData.when_frame_interval?.join('-')})</b></div>
                <div className="row"><span className="mu">WHERE (Lane/BBox):</span> <b>{explainData.where_lane}</b></div>
                <div className="row"><span className="mu">WHICH (Tracks):</span> <b>Track #{explainData.which_track_ids?.join(', #')}</b></div>
                <div className="row"><span className="mu">SCORE:</span> <b>{explainData.score} (Confidence: {explainData.confidence})</b></div>
                <div className="row"><span className="mu">VALIDATION TIER:</span> <span className="badge" style={{ background: '#0EA5E9' }}>{explainData.validation_status}</span></div>

                <h3 style={{ marginTop: '14px' }}>WHY (RULE &amp; FEATURE CONTRIBUTIONS)</h3>
                <pre>{JSON.stringify(explainData.why_rule_contributions, null, 2)}</pre>

                <h3>ROBUST BASELINE STATS</h3>
                <div className="row"><span className="mu">Median:</span> <b>{explainData.baseline?.median}</b></div>
                <div className="row"><span className="mu">MAD:</span> <b>{explainData.baseline?.mad}</b></div>
                <div className="row"><span className="mu">Modified Z-Score:</span> <b>{explainData.baseline?.modified_z_score} MADs</b></div>
              </div>

              <div className="card">
                <h3>EVIDENCE FRAME PACKAGE</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div>
                    <small className="mu">BEFORE FRAME</small>
                    <div style={{ background: '#CBD5E1', borderRadius: '6px', padding: '10px', fontSize: '11px', fontFamily: 'monospace' }}>
                      {explainData.evidence_frames?.before}
                    </div>
                  </div>
                  <div>
                    <small className="mu">EVENT FRAME (PEAK TRIGGER)</small>
                    <div style={{ background: '#EEF2FF', border: '1px solid #C7D2FE', borderRadius: '6px', padding: '10px', fontSize: '11px', fontFamily: 'monospace' }}>
                      {explainData.evidence_frames?.event}
                    </div>
                  </div>
                  <div>
                    <small className="mu">AFTER FRAME</small>
                    <div style={{ background: '#CBD5E1', borderRadius: '6px', padding: '10px', fontSize: '11px', fontFamily: 'monospace' }}>
                      {explainData.evidence_frames?.after}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* PAGE 5: RESEARCH REPORT VIEWER */}
      {subTab === 'report' && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3>AUTOMATED 19-SECTION RESEARCH REPORT GENERATOR</h3>
            <button className="b" onClick={() => fetchReport(selectedExpId)}>
              Refresh Markdown Report
            </button>
          </div>
          <pre style={{ background: '#F8FAFC', color: '#0F172A', border: '1px solid #E2E8F0', whiteSpace: 'pre-wrap', fontSize: '13px', lineHeight: '1.6' }}>
            {reportMarkdown || 'Loading markdown report...'}
          </pre>
        </div>
      )}
    </div>
  );
};
