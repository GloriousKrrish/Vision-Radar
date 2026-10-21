import React, { useState, useEffect } from 'react';
import { apiUrl } from '../config/api';
import { useSharedJob } from '../context/SharedJobContext';

interface ExportPreviewData {
  job_id: number;
  video_filename: string;
  domain: 'traffic' | 'examination';
  status: string;
  progress_pct: number;
  is_complete: boolean;
  recommended_mode: 'final' | 'snapshot';
  counts: {
    tracks: number;
    events: number;
    speed_records: number;
  };
  planned_sheets: string[];
  formats: string[];
}

export const ExcelExportModal: React.FC = () => {
  const { exportModalOpen, exportTargetJobId, closeExportModal, activeJob } = useSharedJob();

  const [mode, setMode] = useState<'final' | 'snapshot'>('final');
  const [preview, setPreview] = useState<ExportPreviewData | null>(null);
  const [loadingPreview, setLoadingPreview] = useState<boolean>(false);
  const [exporting, setExporting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<{ filename: string; sizeBytes: number } | null>(null);

  const jobId = exportTargetJobId || activeJob?.id;

  // Load preview data whenever modal opens for a job
  useEffect(() => {
    if (!exportModalOpen || !jobId) {
      setPreview(null);
      setSuccessInfo(null);
      setErrorMessage(null);
      return;
    }

    setLoadingPreview(true);
    setErrorMessage(null);
    setSuccessInfo(null);

    fetch(apiUrl(`/api/v1/jobs/${jobId}/export/excel/preview`))
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load export preview`);
        return res.json();
      })
      .then((data: ExportPreviewData) => {
        setPreview(data);
        setMode(data.recommended_mode || (data.is_complete ? 'final' : 'snapshot'));
      })
      .catch(err => {
        setErrorMessage(err.message || 'Failed to fetch export preview');
      })
      .finally(() => {
        setLoadingPreview(false);
      });
  }, [exportModalOpen, jobId]);

  if (!exportModalOpen) return null;

  const handleDownload = async () => {
    if (!jobId) return;
    setExporting(true);
    setErrorMessage(null);
    setSuccessInfo(null);

    try {
      const url = apiUrl(`/api/v1/jobs/${jobId}/export/excel?mode=${mode}`);
      const res = await fetch(url);
      if (!res.ok) {
        let errDetail = `HTTP ${res.status}`;
        try {
          const json = await res.json();
          if (json.detail) errDetail = json.detail;
        } catch {
          // ignore
        }
        throw new Error(errDetail);
      }

      // Extract filename from Content-Disposition if present
      let filename = `VisionRadar_Job_${jobId}_${mode}.xlsx`;
      const disposition = res.headers.get('Content-Disposition');
      if (disposition) {
        const match = disposition.match(/filename="?([^";]+)"?/);
        if (match && match[1]) filename = match[1];
      }

      const blob = await res.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(downloadUrl);

      setSuccessInfo({
        filename,
        sizeBytes: blob.size
      });
    } catch (err: any) {
      setErrorMessage(err.message || 'Excel export generation failed. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px'
      }}
      onClick={e => {
        if (e.target === e.currentTarget && !exporting) closeExportModal();
      }}
    >
      <div
        style={{
          background: '#ffffff',
          borderRadius: '14px',
          width: '100%',
          maxWidth: '620px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          border: '1px solid #E2E8F0',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '90vh'
        }}
      >
        {/* Modal Header */}
        <div
          style={{
            background: 'linear-gradient(135deg, #1E293B 0%, #0F172A 100%)',
            color: '#ffffff',
            padding: '18px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '20px' }}>📊</span>
              <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, letterSpacing: '0.2px' }}>
                Enterprise Excel Intelligence Export
              </h2>
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94A3B8' }}>
              Universal Video Intelligence · Standardized Multi-Sheet Workbook (.xlsx)
            </p>
          </div>
          <button
            onClick={closeExportModal}
            disabled={exporting}
            style={{
              background: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              color: '#ffffff',
              borderRadius: '8px',
              width: '32px',
              height: '32px',
              cursor: 'pointer',
              fontSize: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background 0.15s'
            }}
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px 24px', overflowY: 'auto', flex: 1 }}>
          {/* Target Job Summary */}
          <div
            style={{
              background: '#F8FAFC',
              borderRadius: '10px',
              border: '1px solid #E2E8F0',
              padding: '14px 16px',
              marginBottom: '18px',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
              gap: '12px'
            }}
          >
            <div>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>TARGET JOB</div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', marginTop: '2px' }}>
                Job #{jobId || 'N/A'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>DOMAIN PIPELINE</div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: preview?.domain === 'examination' ? '#7C3AED' : '#2563EB', marginTop: '2px' }}>
                {preview?.domain === 'examination' ? '🎓 Examination' : '🚗 Traffic Intel'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>SOURCE VIDEO</div>
              <div style={{ fontSize: '13px', fontWeight: 500, color: '#334155', marginTop: '2px', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                {preview?.video_filename || 'Video Container'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>STATUS</div>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', marginTop: '2px' }}>
                <span
                  style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: preview?.status === 'SUCCEEDED' ? '#10B981' : preview?.status === 'RUNNING' ? '#3B82F6' : '#F59E0B'
                  }}
                />
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#0F172A' }}>
                  {preview?.status || activeJob?.status || 'UNKNOWN'}
                </span>
              </div>
            </div>
          </div>

          {/* Export Mode Selection */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '8px' }}>
              SELECT EXPORT MODE
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div
                onClick={() => setMode('final')}
                style={{
                  border: mode === 'final' ? '2px solid #4F46E5' : '1px solid #CBD5E1',
                  background: mode === 'final' ? '#EEF2FF' : '#ffffff',
                  borderRadius: '10px',
                  padding: '12px 14px',
                  cursor: 'pointer',
                  transition: 'all 0.15s'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input
                    type="radio"
                    name="exportMode"
                    checked={mode === 'final'}
                    onChange={() => setMode('final')}
                    style={{ accentColor: '#4F46E5' }}
                  />
                  <strong style={{ fontSize: '13px', color: '#1E293B' }}>Final Export</strong>
                </div>
                <p style={{ margin: '6px 0 0 20px', fontSize: '11px', color: '#64748B', lineHeight: '1.4' }}>
                  Complete verified intelligence report for finished jobs. Includes all calibrated metrics, events, and data dictionary.
                </p>
              </div>

              <div
                onClick={() => setMode('snapshot')}
                style={{
                  border: mode === 'snapshot' ? '2px solid #4F46E5' : '1px solid #CBD5E1',
                  background: mode === 'snapshot' ? '#EEF2FF' : '#ffffff',
                  borderRadius: '10px',
                  padding: '12px 14px',
                  cursor: 'pointer',
                  transition: 'all 0.15s'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <input
                    type="radio"
                    name="exportMode"
                    checked={mode === 'snapshot'}
                    onChange={() => setMode('snapshot')}
                    style={{ accentColor: '#4F46E5' }}
                  />
                  <strong style={{ fontSize: '13px', color: '#1E293B' }}>Live Snapshot</strong>
                </div>
                <p style={{ margin: '6px 0 0 20px', fontSize: '11px', color: '#64748B', lineHeight: '1.4' }}>
                  In-flight snapshot of records available right now. Non-blocking; video processing continues uninterrupted.
                </p>
              </div>
            </div>
          </div>

          {/* Planned Sheets & Record Counts */}
          {loadingPreview ? (
            <div style={{ textAlign: 'center', padding: '24px 0', color: '#64748B', fontSize: '13px' }}>
              Loading job metadata and sheet structure...
            </div>
          ) : preview ? (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '16px' }}>
                <div style={{ background: '#F1F5F9', padding: '10px 12px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>
                    {preview.domain === 'examination' ? 'Monitored Persons' : 'Unique Tracks'}
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#0F172A', marginTop: '2px' }}>
                    {preview.counts.tracks.toLocaleString()}
                  </div>
                </div>
                <div style={{ background: '#F1F5F9', padding: '10px 12px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>
                    {preview.domain === 'examination' ? 'Behavioral Events' : 'Traffic Violations'}
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#DC2626', marginTop: '2px' }}>
                    {preview.counts.events.toLocaleString()}
                  </div>
                </div>
                <div style={{ background: '#F1F5F9', padding: '10px 12px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>Speed Records</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#2563EB', marginTop: '2px' }}>
                    {preview.counts.speed_records > 0 ? preview.counts.speed_records.toLocaleString() : 'N/A'}
                  </div>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', fontWeight: 700, color: '#64748B', marginBottom: '6px' }}>
                  WORKBOOK SHEETS INCLUDED ({preview.planned_sheets.length} SHEETS)
                </label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {preview.planned_sheets.map((sheet, idx) => (
                    <span
                      key={idx}
                      style={{
                        background: '#E2E8F0',
                        color: '#334155',
                        fontSize: '11px',
                        fontWeight: 600,
                        padding: '3px 8px',
                        borderRadius: '6px'
                      }}
                    >
                      📑 {sheet}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : null}

          {/* Error Banner */}
          {errorMessage && (
            <div
              style={{
                marginTop: '16px',
                background: '#FEE2E2',
                border: '1px solid #FCA5A5',
                borderRadius: '8px',
                padding: '10px 14px',
                color: '#991B1B',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <span>⚠️</span>
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Success Banner */}
          {successInfo && (
            <div
              style={{
                marginTop: '16px',
                background: '#DCFCE7',
                border: '1px solid #86EFAC',
                borderRadius: '8px',
                padding: '12px 14px',
                color: '#166534',
                fontSize: '13px'
              }}
            >
              <div style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span>✅</span>
                <span>Workbook Downloaded Successfully!</span>
              </div>
              <div style={{ fontSize: '11px', color: '#15803D', marginTop: '4px' }}>
                File: <strong>{successInfo.filename}</strong> ({(successInfo.sizeBytes / 1024).toFixed(1)} KB)
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div
          style={{
            background: '#F8FAFC',
            borderTop: '1px solid #E2E8F0',
            padding: '14px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <div style={{ fontSize: '11px', color: '#64748B' }}>
            Format: <strong>openpyxl .xlsx</strong> · Corporate Palette · Auto-filters
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={closeExportModal}
              disabled={exporting}
              style={{
                background: '#ffffff',
                border: '1px solid #CBD5E1',
                borderRadius: '8px',
                padding: '8px 16px',
                fontSize: '13px',
                fontWeight: 600,
                color: '#475569',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
            <button
              onClick={handleDownload}
              disabled={exporting || !jobId}
              style={{
                background: exporting ? '#94A3B8' : '#4F46E5',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 20px',
                fontSize: '13px',
                fontWeight: 600,
                color: '#ffffff',
                cursor: exporting || !jobId ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
              }}
            >
              {exporting ? (
                <>
                  <span className="spinner" style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.6s linear infinite' }} />
                  <span>Generating Excel...</span>
                </>
              ) : (
                <>
                  <span>📥</span>
                  <span>Download .xlsx</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
