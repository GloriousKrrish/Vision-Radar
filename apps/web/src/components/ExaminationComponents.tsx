import React, { useState } from 'react';
import { ExaminationEvent, ExaminationPersonTrack, ExaminationZone, SceneClassificationResult, AnalysisModule, GeminiStatus, GeminiExamAnalysis } from '../types';

// ===========================================================================
// Scene Recommendation Banner
// ===========================================================================
interface SceneBannerProps {
  recommendation: SceneClassificationResult & { video_id: number; video_name: string };
  onAccept: (videoId: number, module: AnalysisModule) => void;
  onSelectModule: (videoId: number, module: AnalysisModule) => void;
  onDismiss: () => void;
}

export const SceneRecommendationBanner: React.FC<SceneBannerProps> = ({
  recommendation,
  onAccept,
  onSelectModule,
  onDismiss
}) => {
  const isTraffic = recommendation.scene_type === 'TRAFFIC';
  const isExam = recommendation.scene_type === 'CLASSROOM_EXAMINATION';

  const badgeBg = isTraffic ? '#065F46' : isExam ? '#3730A3' : '#78350F';
  const badgeColor = isTraffic ? '#34D399' : isExam ? '#A5B4FC' : '#FCD34D';

  return (
    <div style={{
      margin: '12px 0',
      padding: '14px 18px',
      background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96))',
      border: '1px solid #4F46E5',
      borderRadius: '8px',
      boxShadow: '0 4px 20px rgba(79, 70, 229, 0.25)',
      color: '#F8FAFC'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '16px' }}>◎</span>
          <b style={{ fontSize: '13px', color: '#818CF8', letterSpacing: '0.5px' }}>
            UNIVERSAL VIDEO INTELLIGENCE · SCENE ROUTER
          </b>
          <span style={{
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 700,
            background: badgeBg,
            color: badgeColor
          }}>
            {recommendation.scene_type.replace('_', ' ')}
          </span>
          <span style={{ fontSize: '11px', color: '#94A3B8' }}>
            ({Math.round(recommendation.confidence * 100)}% calibrated confidence)
          </span>
        </div>
        <button
          onClick={onDismiss}
          style={{ background: 'transparent', border: 'none', color: '#94A3B8', cursor: 'pointer', fontSize: '14px' }}
          title="Dismiss banner"
        >
          ✕
        </button>
      </div>

      <p style={{ margin: '8px 0 12px 0', fontSize: '12px', color: '#CBD5E1', lineHeight: '1.4' }}>
        {recommendation.reason}
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <button
          onClick={() => onAccept(recommendation.video_id, recommendation.recommended_module)}
          style={{
            background: '#4F46E5',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: '6px',
            padding: '6px 14px',
            fontSize: '12px',
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            transition: 'background 0.15s ease'
          }}
        >
          ✓ Accept Recommendation ({recommendation.recommended_module === 'traffic' ? '🚗 Traffic Pipeline' : '🎓 Examination Pipeline'})
        </button>

        <span style={{ fontSize: '11px', color: '#64748B', margin: '0 4px' }}>or override manually:</span>

        <button
          onClick={() => onSelectModule(recommendation.video_id, 'traffic')}
          style={{
            background: '#1E293B',
            color: '#94A3B8',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '5px 12px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          🚗 Traffic Pipeline
        </button>

        <button
          onClick={() => onSelectModule(recommendation.video_id, 'examination')}
          style={{
            background: '#1E293B',
            color: '#94A3B8',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '5px 12px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          🎓 Examination Pipeline
        </button>
      </div>
    </div>
  );
};

// ===========================================================================
// Primary Finding Panel (Phase 6.A & 6.C)
// ===========================================================================
interface PrimaryFindingProps {
  primaryFinding: ExaminationEvent | null;
  onReview: (eventId: string, status: 'ACCEPTED' | 'REJECTED', notes: string) => void;
  onSeekTime?: (timeSec: number) => void;
  jobId?: number;
}

export const PrimaryFindingPanel: React.FC<PrimaryFindingProps> = ({
  primaryFinding,
  onReview,
  onSeekTime,
  jobId
}) => {
  const [noteDraft, setNoteDraft] = useState('');
  const [downloading, setDownloading] = useState(false);

  const handleDownloadReport = async () => {
    if (!jobId) return;
    try {
      setDownloading(true);
      const res = await fetch(`/api/v1/jobs/${jobId}/examination/report`);
      if (!res.ok) throw new Error('Report fetch failed');
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `examination_intelligence_report_job_${jobId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert('Failed to generate report: ' + e);
    } finally {
      setDownloading(false);
    }
  };

  if (!primaryFinding) {
    return (
      <div className="card" style={{
        marginTop: '16px',
        padding: '16px',
        background: '#0F172A',
        border: '1px solid #1E293B',
        borderRadius: '8px',
        color: '#F8FAFC'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: '#10B981', fontSize: '18px' }}>✓</span>
            <div>
              <b style={{ color: '#F8FAFC', fontSize: '13px' }}>PRIMARY PROCTORING STATUS: NO HIGH-PRIORITY ANOMALIES DETECTED</b>
              <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '2px' }}>
                All monitored student tracks remained within normal behavioral boundaries. Insufficient evidence of electronic device or zone violations.
              </div>
            </div>
          </div>
          {jobId && (
            <button
              onClick={handleDownloadReport}
              disabled={downloading}
              style={{
                background: '#1E293B',
                color: '#CBD5E1',
                border: '1px solid #334155',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              {downloading ? '⏳ Generating...' : '📄 Download Forensic Report'}
            </button>
          )}
        </div>
      </div>
    );
  }

  const isRed = primaryFinding.priority === 'RED';
  const strength = primaryFinding.evidence_strength || 'MEDIUM';
  const isAccepted = primaryFinding.review_status === 'ACCEPTED';
  const isRejected = primaryFinding.review_status === 'REJECTED';
  const suppFrames = primaryFinding.supporting_frames || [primaryFinding.start_frame];

  return (
    <div className="card" style={{
      marginTop: '16px',
      padding: '16px 20px',
      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(15, 23, 42, 0.98))',
      border: `2px solid ${isRed ? '#EF4444' : '#F59E0B'}`,
      borderRadius: '8px',
      boxShadow: isRed ? '0 4px 25px rgba(239, 68, 68, 0.25)' : '0 4px 20px rgba(245, 158, 11, 0.20)',
      color: '#F8FAFC'
    }}>
      {/* Header Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{
            background: isRed ? '#EF4444' : '#F59E0B',
            color: '#FFFFFF',
            padding: '4px 10px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 800,
            letterSpacing: '0.5px'
          }}>
            {isRed ? '🚨 URGENT PROCTOR REVIEW REQUIRED' : '⚠️ CANDIDATE VISUAL EVENT'}
          </span>

          <span style={{
            padding: '3px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 700,
            background: strength === 'HIGH' ? '#065F46' : '#78350F',
            color: strength === 'HIGH' ? '#34D399' : '#FCD34D'
          }}>
            EVIDENCE: {strength}
          </span>

          <span style={{ fontSize: '11px', color: '#94A3B8' }}>
            (Heuristic Score: {Math.round((primaryFinding.event_confidence || primaryFinding.confidence) * 100)}%)
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{
            padding: '3px 8px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: 700,
            background: isAccepted ? '#065F46' : isRejected ? '#7F1D1D' : '#78350F',
            color: isAccepted ? '#34D399' : isRejected ? '#FCA5A5' : '#FCD34D'
          }}>
            STATUS: {primaryFinding.review_status}
          </span>
          {jobId && (
            <button
              onClick={handleDownloadReport}
              disabled={downloading}
              style={{
                background: '#4F46E5',
                color: '#FFFFFF',
                border: 'none',
                padding: '5px 12px',
                borderRadius: '5px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              {downloading ? '⏳ Exporting...' : '📄 Download Forensic Report'}
            </button>
          )}
        </div>
      </div>

      {/* Main Finding Overview */}
      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr auto', gap: '16px', alignItems: 'start' }}>
        <div>
          <div style={{ fontSize: '16px', fontWeight: 800, color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>{primaryFinding.event_type === 'POSSIBLE_PHONE_INTERACTION' ? '📱' : '⚠️'}</span>
            <span>{primaryFinding.event_type.replace(/_/g, ' ')}</span>
            <span style={{ color: '#818CF8', fontSize: '14px', fontWeight: 600 }}>· Person Track #{primaryFinding.person_track_id}</span>
          </div>

          <div style={{ marginTop: '6px', fontSize: '12px', color: '#CBD5E1', lineHeight: '1.5' }}>
            <b>Observation:</b> {primaryFinding.supporting_observations || primaryFinding.description}
          </div>

          <div style={{ marginTop: '6px', display: 'flex', gap: '16px', fontSize: '11px', color: '#94A3B8', flexWrap: 'wrap' }}>
            <span>Observation Window: <b style={{ color: '#F1F5F9' }}>{primaryFinding.start_time.toFixed(1)}s – {(primaryFinding.end_time || primaryFinding.start_time).toFixed(1)}s</b></span>
            <span>Frames: <b style={{ color: '#F1F5F9' }}>#{primaryFinding.start_frame} – #{primaryFinding.end_frame || primaryFinding.start_frame}</b></span>
            <span>Detection Confidence: <b style={{ color: '#34D399' }}>{Math.round((primaryFinding.detection_confidence || primaryFinding.confidence) * 100)}%</b></span>
            <span>Association Confidence: <b style={{ color: '#818CF8' }}>{Math.round((primaryFinding.association_confidence || 0.8) * 100)}%</b></span>
          </div>

          {/* Visibility & Uncertainty Warnings */}
          {((primaryFinding.visibility_warnings && primaryFinding.visibility_warnings.length > 0) ||
            (primaryFinding.uncertainty_reasons && primaryFinding.uncertainty_reasons.length > 0)) && (
            <div style={{ marginTop: '8px', padding: '6px 10px', background: 'rgba(245, 158, 11, 0.10)', borderLeft: '3px solid #F59E0B', borderRadius: '4px', fontSize: '11px', color: '#FCD34D' }}>
              <b>Confounding Factors / Measurement Uncertainty:</b>{' '}
              {[...(primaryFinding.visibility_warnings || []), ...(primaryFinding.uncertainty_reasons || [])].join('; ')}
            </div>
          )}

          {/* Evidence Frame Timeline / Click-to-Seek */}
          {suppFrames.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#94A3B8', marginRight: '8px' }}>
                Keyframe Evidence Timeline ({suppFrames.length} frames):
              </span>
              <div style={{ display: 'inline-flex', gap: '5px', flexWrap: 'wrap', verticalAlign: 'middle' }}>
                {suppFrames.map(fIdx => {
                  const timeApprox = (fIdx / 30.0);
                  return (
                    <button
                      key={fIdx}
                      onClick={() => onSeekTime && onSeekTime(timeApprox)}
                      style={{
                        background: '#1E293B',
                        color: '#A5B4FC',
                        border: '1px solid #4F46E5',
                        borderRadius: '4px',
                        padding: '2px 8px',
                        fontSize: '10px',
                        fontWeight: 700,
                        cursor: 'pointer'
                      }}
                      title={`Jump to Frame #${fIdx} (${timeApprox.toFixed(1)}s)`}
                    >
                      f#{fIdx} ({timeApprox.toFixed(1)}s) 🔍
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Proctor Review Actions */}
        <div style={{
          minWidth: '220px',
          background: '#0F172A',
          padding: '12px',
          borderRadius: '6px',
          border: '1px solid #334155'
        }}>
          <div style={{ fontSize: '10px', color: '#94A3B8', fontWeight: 700, marginBottom: '6px' }}>
            PROCTOR DECISION AUDIT
          </div>
          <input
            type="text"
            placeholder="Reviewer notes..."
            value={noteDraft || primaryFinding.reviewer_notes || ''}
            onChange={e => setNoteDraft(e.target.value)}
            style={{
              width: '100%',
              padding: '5px 8px',
              fontSize: '11px',
              background: '#1E293B',
              color: '#F8FAFC',
              border: '1px solid #475569',
              borderRadius: '4px',
              marginBottom: '8px',
              boxSizing: 'border-box'
            }}
          />
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={() => onReview(primaryFinding.event_id, 'ACCEPTED', noteDraft || primaryFinding.reviewer_notes || 'Verified by proctor')}
              style={{
                flex: 1,
                background: '#059669',
                color: '#FFF',
                border: 'none',
                borderRadius: '4px',
                padding: '6px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
              title="Confirm this candidate event as verified evidence"
            >
              ✓ Confirm Evidence
            </button>
            <button
              onClick={() => onReview(primaryFinding.event_id, 'REJECTED', noteDraft || primaryFinding.reviewer_notes || 'Dismissed by proctor')}
              style={{
                flex: 1,
                background: '#DC2626',
                color: '#FFF',
                border: 'none',
                borderRadius: '4px',
                padding: '6px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
              title="Dismiss as non-actionable or false positive"
            >
              ✗ Dismiss
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ===========================================================================
// Gemini Multimodal Intelligence Contextual Panel
// ===========================================================================
interface GeminiContextualProps {
  jobId: number | null;
  geminiStatus: GeminiStatus | null;
  geminiAnalysis: GeminiExamAnalysis | null;
  isAnalyzing: boolean;
  onAnalyze: (forceRefresh?: boolean) => void;
  onSeekTime?: (timeSec: number) => void;
  errorDetail?: string | null;
}

export const GeminiContextualPanel: React.FC<GeminiContextualProps> = ({
  jobId,
  geminiStatus,
  geminiAnalysis,
  isAnalyzing,
  onAnalyze,
  onSeekTime,
  errorDetail
}) => {
  const isOnline = geminiStatus?.status === 'READY';
  const isMissingKey = geminiStatus?.status === 'KEY_MISSING';

  return (
    <div className="card" style={{
      marginTop: '16px',
      padding: '16px 20px',
      background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(24, 18, 43, 0.98))',
      border: '1px solid #4F46E5',
      borderRadius: '8px',
      boxShadow: '0 4px 20px rgba(79, 70, 229, 0.15)',
      color: '#F8FAFC'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '18px' }}>✨</span>
          <div>
            <div style={{ fontSize: '14px', fontWeight: 800, color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Google Gemini Multimodal Contextual Intelligence</span>
              <span style={{
                background: isOnline ? '#065F46' : isMissingKey ? '#78350F' : '#334155',
                color: isOnline ? '#34D399' : isMissingKey ? '#FCD34D' : '#94A3B8',
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: 700
              }}>
                {isOnline ? `ONLINE · ${geminiStatus?.model}` : (geminiStatus?.status || 'CHECKING...')}
              </span>
            </div>
            <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '2px' }}>
              Multimodal contextual reasoning across timestamped keyframes · Complements local ByteTrack and YOLOX-Nano
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {isOnline && (
            <button
              onClick={() => onAnalyze(Boolean(geminiAnalysis))}
              disabled={isAnalyzing || !jobId}
              style={{
                background: isAnalyzing ? '#312E81' : '#4F46E5',
                color: '#FFFFFF',
                border: 'none',
                padding: '7px 14px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: isAnalyzing || !jobId ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              {isAnalyzing ? '⏳ Analyzing Keyframes...' : (geminiAnalysis ? '🔄 Re-Analyze with Gemini' : '✨ Analyze with Gemini')}
            </button>
          )}
        </div>
      </div>

      {/* Missing Key Warning */}
      {isMissingKey && (
        <div style={{
          marginTop: '12px',
          padding: '10px 14px',
          background: 'rgba(245, 158, 11, 0.1)',
          borderLeft: '4px solid #F59E0B',
          borderRadius: '4px',
          fontSize: '11px',
          color: '#FCD34D'
        }}>
          <b>Gemini API Key Required:</b> Set <code>GEMINI_API_KEY</code> in the server <code>.env</code> file to enable multimodal reasoning and automated report summaries. VisionRadar's local computer vision detection and tracking remain fully operational.
        </div>
      )}

      {/* Error Detail */}
      {errorDetail && (
        <div style={{
          marginTop: '12px',
          padding: '10px 14px',
          background: 'rgba(239, 68, 68, 0.1)',
          borderLeft: '4px solid #EF4444',
          borderRadius: '4px',
          fontSize: '11px',
          color: '#FCA5A5'
        }}>
          <b>Analysis Error:</b> {errorDetail}
        </div>
      )}

      {/* Analysis Output Body */}
      {geminiAnalysis && (
        <div style={{ marginTop: '14px', borderTop: '1px solid #1E293B', paddingTop: '12px' }}>
          {/* Scene Summary */}
          <div style={{ background: '#0F172A', padding: '12px 14px', borderRadius: '6px', border: '1px solid #334155' }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: '#818CF8', letterSpacing: '0.5px', marginBottom: '4px' }}>
              SCENE CONTEXTUAL SUMMARY (Model: {geminiAnalysis.model_identifier})
            </div>
            <div style={{ fontSize: '12px', color: '#E2E8F0', lineHeight: '1.6' }}>
              {geminiAnalysis.scene_summary}
            </div>
          </div>

          {/* Candidate Findings Multimodal Interpretation */}
          {geminiAnalysis.candidate_findings && geminiAnalysis.candidate_findings.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ fontSize: '11px', fontWeight: 800, color: '#94A3B8', marginBottom: '8px' }}>
                EVIDENCE-LINKED CANDIDATE INTERPRETATIONS:
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {geminiAnalysis.candidate_findings.map((finding, idx) => (
                  <div key={idx} style={{
                    background: '#1E293B',
                    padding: '12px',
                    borderRadius: '6px',
                    borderLeft: `4px solid ${finding.recommended_review_priority === 'RED' ? '#EF4444' : '#F59E0B'}`
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ fontSize: '12px', fontWeight: 800, color: '#F8FAFC' }}>
                        <span>🔍 {finding.event_type.replace(/_/g, ' ')}</span>
                        <span style={{ fontSize: '11px', color: '#94A3B8', marginLeft: '8px' }}>(Event ID: {finding.event_id})</span>
                      </div>
                      <span style={{
                        background: finding.recommended_review_priority === 'RED' ? '#7F1D1D' : '#78350F',
                        color: finding.recommended_review_priority === 'RED' ? '#FCA5A5' : '#FCD34D',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '10px',
                        fontWeight: 700
                      }}>
                        PRIORITY: {finding.recommended_review_priority}
                      </span>
                    </div>

                    <div style={{ marginTop: '6px', fontSize: '12px', color: '#CBD5E1', lineHeight: '1.5' }}>
                      <b>Observable Behaviour:</b> {finding.observed_behaviour}
                    </div>

                    {finding.contextual_assessment && (
                      <div style={{ marginTop: '4px', fontSize: '11px', color: '#A5B4FC', lineHeight: '1.5' }}>
                        <b>Contextual Assessment:</b> {finding.contextual_assessment}
                      </div>
                    )}

                    {finding.supporting_frame_references && finding.supporting_frame_references.length > 0 && (
                      <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontSize: '10px', color: '#94A3B8', fontWeight: 700 }}>Supporting Frames:</span>
                        {finding.supporting_frame_references.map(fIdx => (
                          <button
                            key={fIdx}
                            onClick={() => onSeekTime && onSeekTime(fIdx / 30.0)}
                            style={{
                              background: '#0F172A',
                              color: '#818CF8',
                              border: '1px solid #3730A3',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              fontSize: '10px',
                              fontWeight: 700,
                              cursor: 'pointer'
                            }}
                          >
                            #{fIdx} ({(fIdx / 30.0).toFixed(1)}s)
                          </button>
                        ))}
                      </div>
                    )}

                    {((finding.uncertainty_notes && finding.uncertainty_notes.length > 0) ||
                      (finding.visibility_warnings && finding.visibility_warnings.length > 0)) && (
                      <div style={{ marginTop: '6px', fontSize: '10px', color: '#FCD34D' }}>
                        <b>Limitations:</b> {[...(finding.uncertainty_notes || []), ...(finding.visibility_warnings || [])].join('; ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Contradictory Evidence Notes */}
          {geminiAnalysis.contradictory_evidence_notes && geminiAnalysis.contradictory_evidence_notes.length > 0 && (
            <div style={{
              marginTop: '12px',
              padding: '8px 12px',
              background: 'rgba(5, 150, 105, 0.1)',
              borderLeft: '3px solid #059669',
              borderRadius: '4px',
              fontSize: '11px',
              color: '#34D399'
            }}>
              <b>Observations Weakening Misconduct Claim:</b> {geminiAnalysis.contradictory_evidence_notes.join('; ')}
            </div>
          )}

          {/* Privacy & Ethical Notice */}
          <div style={{ marginTop: '10px', fontSize: '10px', color: '#64748B', display: 'flex', justifyContent: 'space-between' }}>
            <span>Proctor Recommendation: <b style={{ color: '#E2E8F0' }}>{geminiAnalysis.proctor_action_recommendation}</b></span>
            <span>Privacy Note: 6 keyframes sampled · Continuous raw video never uploaded</span>
          </div>
        </div>
      )}
    </div>
  );
};

// ===========================================================================
// Examination Overview Card
// ===========================================================================
interface ExamOverviewProps {
  totalPersons: number;
  totalEvents: number;
  pendingReviews: number;
  acceptedReviews: number;
  activeZones: number;
  pipelineStatus: string;
}

export const ExaminationOverviewCard: React.FC<ExamOverviewProps> = ({
  totalPersons,
  totalEvents,
  pendingReviews,
  acceptedReviews,
  activeZones,
  pipelineStatus
}) => {
  return (
    <div className="card" style={{ marginTop: '16px' }}>
      <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>🎓 Examination &amp; Proctoring Intelligence</span>
          <span className="badge" style={{ background: '#EEF2FF', color: '#4F46E5', fontWeight: 700 }}>
            DOMAIN: CLASSROOM / EXAMINATION
          </span>
        </div>
        <span className="badge" style={{
          background: pendingReviews > 0 ? '#FEF3C7' : '#D1FAE5',
          color: pendingReviews > 0 ? '#92400E' : '#065F46',
          fontWeight: 700, fontSize: '12px'
        }}>
          {pendingReviews > 0 ? `⚠️ ${pendingReviews} EVENTS REQUIRE HUMAN REVIEW` : '✓ PROCTOR REVIEW UP TO DATE'}
        </span>
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', margin: '12px 0' }}>
        {[
          { label: 'MONITORED PERSONS', value: totalPersons, unit: 'students/staff', color: '#1E293B' },
          { label: 'CANDIDATE EVENTS', value: totalEvents, unit: 'observable events', color: '#4F46E5' },
          { label: 'PENDING REVIEW', value: pendingReviews, unit: 'unreviewed', color: '#D97706' },
          { label: 'VERIFIED EVIDENCE', value: acceptedReviews, unit: 'accepted', color: '#059669' },
        ].map((tile, i) => (
          <div key={i} style={{ background: '#F8FAFC', padding: '10px 14px', borderRadius: '6px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontSize: '10px', color: '#64748B', fontWeight: 700 }}>{tile.label}</div>
            <div style={{ fontSize: '20px', fontWeight: 800, color: tile.color, marginTop: '2px' }}>
              {tile.value} <span style={{ fontSize: '11px', fontWeight: 500, color: '#64748B' }}>{tile.unit}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#64748B', paddingTop: '8px', borderTop: '1px solid #E2E8F0' }}>
        <span>Configured Seating Zones: <b style={{ color: '#4F46E5' }}>{activeZones} zones</b></span>
        <span>Ethical AI Standard: <b style={{ color: '#059669' }}>Proctor-in-the-Loop (Zero Automated Accusations)</b></span>
        <span>Pipeline Status: <b style={{ color: '#10B981' }}>{pipelineStatus}</b></span>
      </div>
    </div>
  );
};

// ===========================================================================
// Examination Candidate Events & Proctor Review Log Table (Phase 6.B, 6.C, 6.F)
// ===========================================================================
interface ExamTableProps {
  events: ExaminationEvent[];
  onReview: (eventId: string, status: 'ACCEPTED' | 'REJECTED', notes: string) => void;
  onInspectPerson: (personTrackId: number) => void;
  selectedPersonId: number | null;
  onSeekTime?: (timeSec: number) => void;
}

export const ExaminationEventsTable: React.FC<ExamTableProps> = ({
  events,
  onReview,
  onInspectPerson,
  selectedPersonId,
  onSeekTime
}) => {
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'RED_PRIORITY' | 'PENDING' | 'ACCEPTED' | 'REJECTED'>('ALL');
  const [typeFilter, setTypeFilter] = useState<string>('ALL');
  const [noteDrafts, setNoteDrafts] = useState<{ [id: string]: string }>({});
  const [page, setPage] = useState<number>(1);
  const PAGE_SIZE = 8;

  const filtered = events.filter(e => {
    if (statusFilter === 'RED_PRIORITY' && e.priority !== 'RED') return false;
    if (statusFilter === 'PENDING' && e.review_status !== 'PENDING') return false;
    if (statusFilter === 'ACCEPTED' && e.review_status !== 'ACCEPTED') return false;
    if (statusFilter === 'REJECTED' && e.review_status !== 'REJECTED') return false;
    if (typeFilter !== 'ALL' && e.event_type !== typeFilter) return false;
    return true;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageData = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const formatEventType = (t: string) => {
    switch (t) {
      case 'POSSIBLE_PHONE_INTERACTION': return '📱 Possible Phone Interaction';
      case 'RESTRICTED_ZONE_ENTRY': return '⛔ Restricted Zone Entry';
      case 'LEAVING_SEATING_ZONE': return '🚶 Leaving Seating Zone';
      case 'POSSIBLE_OBJECT_TRANSFER': return '🤝 Possible Object Transfer';
      case 'REPEATED_ORIENTATION_PATTERN': return '🔄 Repeated Orientation Pattern';
      case 'OTHER_OBSERVABLE_INTERACTION': return '📋 Other Observable Interaction';
      default: return t;
    }
  };

  return (
    <div className="card" style={{ marginTop: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h3 style={{ margin: 0 }}>Candidate Examination Events &amp; Proctor Review Log</h3>
          <div style={{ fontSize: '11px', color: '#64748B', marginTop: '2px' }}>
            Empirically recorded behavioral anomalies. Neutral observable classifications requiring authorized human proctor sign-off.
          </div>
        </div>

        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            value={typeFilter}
            onChange={e => { setTypeFilter(e.target.value); setPage(1); }}
            style={{
              padding: '4px 8px',
              fontSize: '11px',
              borderRadius: '4px',
              border: '1px solid #CBD5E1',
              background: '#F8FAFC',
              color: '#334155'
            }}
          >
            <option value="ALL">All Event Types</option>
            <option value="POSSIBLE_PHONE_INTERACTION">📱 Phone Interaction</option>
            <option value="RESTRICTED_ZONE_ENTRY">⛔ Restricted Zone</option>
            <option value="LEAVING_SEATING_ZONE">🚶 Leaving Seat</option>
            <option value="POSSIBLE_OBJECT_TRANSFER">🤝 Object Transfer</option>
          </select>

          {(['ALL', 'RED_PRIORITY', 'PENDING', 'ACCEPTED', 'REJECTED'] as const).map(f => (
            <button
              key={f}
              onClick={() => { setStatusFilter(f); setPage(1); }}
              style={{
                background: statusFilter === f ? (f === 'RED_PRIORITY' ? '#DC2626' : '#4F46E5') : '#F1F5F9',
                color: statusFilter === f ? '#FFFFFF' : (f === 'RED_PRIORITY' ? '#DC2626' : '#475569'),
                border: f === 'RED_PRIORITY' && statusFilter !== f ? '1px solid #FCA5A5' : 'none',
                borderRadius: '4px',
                padding: '4px 10px',
                fontSize: '11px',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              {f === 'RED_PRIORITY' ? '🚨 RED PRIORITY' : f} ({events.filter(e => {
                if (f === 'RED_PRIORITY') return e.priority === 'RED';
                if (f === 'ALL') return true;
                return e.review_status === f;
              }).length})
            </button>
          ))}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
          <thead>
            <tr style={{ background: '#F8FAFC', borderBottom: '1px solid #E2E8F0', textAlign: 'left', color: '#64748B' }}>
              <th style={{ padding: '8px' }}>PRIORITY</th>
              <th style={{ padding: '8px' }}>EVENT CLASSIFICATION</th>
              <th style={{ padding: '8px' }}>PERSON TRACK</th>
              <th style={{ padding: '8px' }}>TIME / FRAMES</th>
              <th style={{ padding: '8px' }}>EVIDENCE STRENGTH</th>
              <th style={{ padding: '8px' }}>SUPPORTING OBSERVATIONS</th>
              <th style={{ padding: '8px' }}>REVIEW STATUS</th>
              <th style={{ padding: '8px', textAlign: 'right' }}>PROCTOR ACTIONS</th>
            </tr>
          </thead>
          <tbody>
            {pageData.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ padding: '24px', textAlign: 'center', color: '#94A3B8' }}>
                  No examination candidate events recorded for this selection.
                </td>
              </tr>
            ) : (
              pageData.map(evt => {
                const isPending = evt.review_status === 'PENDING';
                const isAccepted = evt.review_status === 'ACCEPTED';
                const isRejected = evt.review_status === 'REJECTED';
                const isSelected = selectedPersonId === evt.person_track_id;
                const isRed = evt.priority === 'RED';
                const strength = evt.evidence_strength || 'MEDIUM';

                return (
                  <tr
                    key={evt.event_id}
                    style={{
                      borderBottom: '1px solid #F1F5F9',
                      background: isSelected ? '#EEF2FF' : isRed ? 'rgba(254, 242, 242, 0.5)' : 'transparent',
                      cursor: 'pointer'
                    }}
                    onClick={() => onInspectPerson(evt.person_track_id)}
                  >
                    <td style={{ padding: '8px' }}>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        fontSize: '9px',
                        fontWeight: 800,
                        background: isRed ? '#FEE2E2' : '#FEF3C7',
                        color: isRed ? '#DC2626' : '#B45309'
                      }}>
                        {evt.priority || 'AMBER'}
                      </span>
                    </td>
                    <td style={{ padding: '8px', fontWeight: 600, color: '#1E293B' }}>
                      {formatEventType(evt.event_type)}
                    </td>
                    <td style={{ padding: '8px' }}>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: '#EEF2FF',
                        color: '#4F46E5',
                        fontWeight: 700
                      }}>
                        Person #{evt.person_track_id}
                      </span>
                    </td>
                    <td style={{ padding: '8px', color: '#334155' }}>
                      {evt.start_time.toFixed(1)}s–{(evt.end_time || evt.start_time).toFixed(1)}s
                      <div style={{ fontSize: '9px', color: '#94A3B8' }}>
                        f#{evt.start_frame}–f#{evt.end_frame || evt.start_frame}
                      </div>
                    </td>
                    <td style={{ padding: '8px' }}>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        fontSize: '10px',
                        fontWeight: 700,
                        background: strength === 'HIGH' ? '#D1FAE5' : '#FEF3C7',
                        color: strength === 'HIGH' ? '#065F46' : '#92400E'
                      }}>
                        {strength}
                      </span>
                      <div style={{ fontSize: '9px', color: '#64748B', marginTop: '2px' }}>
                        Score: {Math.round((evt.event_confidence || evt.confidence) * 100)}%
                      </div>
                    </td>
                    <td style={{ padding: '8px', color: '#475569', maxWidth: '280px' }}>
                      <div style={{ fontSize: '10px', lineHeight: '1.4' }}>
                        {evt.supporting_observations || evt.description}
                      </div>
                      {evt.supporting_frames && evt.supporting_frames.length > 0 && (
                        <div style={{ marginTop: '4px', display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                          {evt.supporting_frames.slice(0, 5).map(f => (
                            <button
                              key={f}
                              onClick={(e) => {
                                e.stopPropagation();
                                if (onSeekTime) onSeekTime(f / 30.0);
                              }}
                              style={{
                                background: '#F1F5F9',
                                border: '1px solid #CBD5E1',
                                borderRadius: '2px',
                                padding: '1px 4px',
                                fontSize: '9px',
                                cursor: 'pointer',
                                color: '#4F46E5'
                              }}
                              title={`Seek to frame #${f}`}
                            >
                              f#{f}
                            </button>
                          ))}
                          {evt.supporting_frames.length > 5 && (
                            <span style={{ fontSize: '9px', color: '#94A3B8' }}>+{evt.supporting_frames.length - 5} more</span>
                          )}
                        </div>
                      )}
                    </td>
                    <td style={{ padding: '8px' }}>
                      <span style={{
                        fontSize: '10px',
                        fontWeight: 700,
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: isAccepted ? '#DCFCE7' : isRejected ? '#FEE2E2' : '#FEF3C7',
                        color: isAccepted ? '#166534' : isRejected ? '#991B1B' : '#92400E'
                      }}>
                        {evt.review_status}
                      </span>
                    </td>
                    <td style={{ padding: '8px', textAlign: 'right' }} onClick={e => e.stopPropagation()}>
                      <div style={{ display: 'flex', gap: '4px', justifyContent: 'flex-end', alignItems: 'center' }}>
                        <input
                          type="text"
                          placeholder="Proctor note..."
                          value={noteDrafts[evt.event_id] ?? evt.reviewer_notes ?? ''}
                          onChange={e => setNoteDrafts({ ...noteDrafts, [evt.event_id]: e.target.value })}
                          style={{
                            padding: '2px 6px',
                            fontSize: '10px',
                            borderRadius: '3px',
                            border: '1px solid #CBD5E1',
                            width: '100px'
                          }}
                        />
                        <button
                          onClick={() => onReview(evt.event_id, 'ACCEPTED', noteDrafts[evt.event_id] || '')}
                          style={{
                            background: '#059669',
                            color: '#FFFFFF',
                            border: 'none',
                            borderRadius: '3px',
                            padding: '3px 6px',
                            fontSize: '10px',
                            fontWeight: 700,
                            cursor: 'pointer'
                          }}
                          title="Confirm candidate event as verified"
                        >
                          ✓
                        </button>
                        <button
                          onClick={() => onReview(evt.event_id, 'REJECTED', noteDrafts[evt.event_id] || '')}
                          style={{
                            background: '#DC2626',
                            color: '#FFFFFF',
                            border: 'none',
                            borderRadius: '3px',
                            padding: '3px 6px',
                            fontSize: '10px',
                            fontWeight: 700,
                            cursor: 'pointer'
                          }}
                          title="Dismiss candidate event as false positive"
                        >
                          ✗
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {filtered.length > 0 && (
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
            Showing <b>{(currentPage - 1) * PAGE_SIZE + 1}</b>–<b>{Math.min(currentPage * PAGE_SIZE, filtered.length)}</b> of <b>{filtered.length}</b> candidate events
          </div>
          <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
            <button
              className="b g"
              style={{ padding: '3px 8px', fontSize: '11px' }}
              disabled={currentPage <= 1}
              onClick={() => setPage(p => Math.max(1, p - 1))}
            >
              ◀ Prev
            </button>
            <span>Page <b>{currentPage}</b> of <b>{totalPages}</b></span>
            <button
              className="b g"
              style={{ padding: '3px 8px', fontSize: '11px' }}
              disabled={currentPage >= totalPages}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            >
              Next ▶
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ===========================================================================
// Examination Sidebar Inspector (Phase 6.C, 6.E)
// ===========================================================================
interface ExamSidebarProps {
  events: ExaminationEvent[];
  selectedPersonId: number | null;
  personTracks: ExaminationPersonTrack[];
  zones: ExaminationZone[];
  onReview: (eventId: string, status: 'ACCEPTED' | 'REJECTED', notes: string) => void;
  onSeekTime?: (timeSec: number) => void;
}

export const ExaminationSidebarInspector: React.FC<ExamSidebarProps> = ({
  events,
  selectedPersonId,
  personTracks,
  zones,
  onReview,
  onSeekTime
}) => {
  const [tab, setTab] = useState<'events' | 'person'>('events');

  const selectedPerson = personTracks.find(t => t.track_id === selectedPersonId);
  const personEvents = events.filter(e => selectedPersonId !== null && e.person_track_id === selectedPersonId);

  // Separate red priority alerts from other events
  const redAlerts = events.filter(e => e.priority === 'RED');
  const otherAlerts = events.filter(e => e.priority !== 'RED');

  return (
    <div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12, borderBottom: '1px solid #1E293B', paddingBottom: 8 }}>
        <button
          className={`tab-pill ${tab === 'events' ? 'active' : 'inactive'}`}
          onClick={() => setTab('events')}
        >
          EVENT ALERTS ({events.length})
        </button>
        <button
          className={`tab-pill ${tab === 'person' ? 'active' : 'inactive'}`}
          onClick={() => setTab('person')}
        >
          PERSON INSPECTOR {selectedPersonId !== null ? `(#${selectedPersonId})` : ''}
        </button>
      </div>

      {tab === 'events' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '520px', overflowY: 'auto' }}>
          {events.length === 0 ? (
            <div style={{ padding: '16px', textAlign: 'center', color: '#64748B', fontSize: '12px' }}>
              No candidate alerts detected in examination stream.
            </div>
          ) : (
            <>
              {redAlerts.length > 0 && (
                <div style={{ fontSize: '10px', fontWeight: 800, color: '#EF4444', letterSpacing: '0.5px', marginTop: '4px' }}>
                  🚨 RED PRIORITY ALERTS ({redAlerts.length})
                </div>
              )}
              {redAlerts.map(e => (
                <div
                  key={e.event_id}
                  style={{
                    background: '#1E293B',
                    border: '1px solid #EF4444',
                    borderLeft: '4px solid #EF4444',
                    borderRadius: '6px',
                    padding: '10px',
                    fontSize: '11px',
                    color: '#F8FAFC'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <b style={{ color: '#F87171' }}>Person #{e.person_track_id} · {e.event_type.replace(/_/g, ' ')}</b>
                    <span style={{
                      fontSize: '9px',
                      fontWeight: 700,
                      padding: '1px 5px',
                      borderRadius: '3px',
                      background: e.review_status === 'ACCEPTED' ? '#065F46' : e.review_status === 'REJECTED' ? '#7F1D1D' : '#78350F',
                      color: e.review_status === 'ACCEPTED' ? '#34D399' : e.review_status === 'REJECTED' ? '#FCA5A5' : '#FCD34D'
                    }}>
                      {e.review_status}
                    </span>
                  </div>

                  <div style={{ color: '#CBD5E1', marginTop: '4px', fontSize: '10px', lineHeight: '1.4' }}>
                    {e.supporting_observations || e.description}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '10px', color: '#94A3B8' }}>
                    <span>Time: <b>{e.start_time.toFixed(1)}s</b></span>
                    <span>Evidence: <b style={{ color: '#34D399' }}>{e.evidence_strength || 'HIGH'}</b></span>
                  </div>

                  {e.supporting_frames && e.supporting_frames.length > 0 && (
                    <div style={{ marginTop: '6px', display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                      {e.supporting_frames.slice(0, 4).map(f => (
                        <button
                          key={f}
                          onClick={() => onSeekTime && onSeekTime(f / 30.0)}
                          style={{
                            background: '#0F172A',
                            color: '#A5B4FC',
                            border: '1px solid #334155',
                            borderRadius: '2px',
                            padding: '1px 4px',
                            fontSize: '9px',
                            cursor: 'pointer'
                          }}
                        >
                          f#{f}
                        </button>
                      ))}
                    </div>
                  )}

                  {e.review_status === 'PENDING' && (
                    <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
                      <button
                        onClick={() => onReview(e.event_id, 'ACCEPTED', 'Verified by proctor')}
                        style={{
                          flex: 1,
                          background: '#059669',
                          color: '#FFF',
                          border: 'none',
                          borderRadius: '3px',
                          padding: '4px',
                          fontSize: '10px',
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                      >
                        ✓ Accept
                      </button>
                      <button
                        onClick={() => onReview(e.event_id, 'REJECTED', 'Dismissed by proctor')}
                        style={{
                          flex: 1,
                          background: '#DC2626',
                          color: '#FFF',
                          border: 'none',
                          borderRadius: '3px',
                          padding: '4px',
                          fontSize: '10px',
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                      >
                        ✗ Dismiss
                      </button>
                    </div>
                  )}
                </div>
              ))}

              {otherAlerts.length > 0 && (
                <div style={{ fontSize: '10px', fontWeight: 700, color: '#94A3B8', letterSpacing: '0.5px', marginTop: '6px' }}>
                  OBSERVABLE EVENTS ({otherAlerts.length})
                </div>
              )}
              {otherAlerts.map(e => (
                <div
                  key={e.event_id}
                  style={{
                    background: '#1E293B',
                    border: `1px solid ${e.review_status === 'ACCEPTED' ? '#059669' : e.review_status === 'REJECTED' ? '#DC2626' : '#334155'}`,
                    borderRadius: '6px',
                    padding: '8px',
                    fontSize: '11px',
                    color: '#F8FAFC'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <b style={{ color: '#818CF8' }}>Person #{e.person_track_id}</b>
                    <span style={{
                      fontSize: '9px',
                      fontWeight: 700,
                      padding: '1px 5px',
                      borderRadius: '3px',
                      background: e.review_status === 'ACCEPTED' ? '#065F46' : e.review_status === 'REJECTED' ? '#7F1D1D' : '#78350F',
                      color: e.review_status === 'ACCEPTED' ? '#34D399' : e.review_status === 'REJECTED' ? '#FCA5A5' : '#FCD34D'
                    }}>
                      {e.review_status}
                    </span>
                  </div>
                  <div style={{ fontWeight: 600, marginTop: '2px', color: '#E2E8F0', fontSize: '10px' }}>
                    {e.event_type.replace(/_/g, ' ')}
                  </div>
                  <div style={{ color: '#94A3B8', marginTop: '2px', fontSize: '10px' }}>
                    {e.description}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      ) : (
        <div>
          {selectedPerson ? (
            <div style={{ fontSize: '12px', color: '#CBD5E1' }}>
              <div style={{ padding: '10px', background: '#1E293B', borderRadius: '6px', marginBottom: '8px' }}>
                <div style={{ fontSize: '13px', fontWeight: 800, color: '#818CF8' }}>
                  Person Track #{selectedPerson.track_id}
                </div>
                <div style={{ marginTop: '4px', fontSize: '11px' }}>
                  Assigned Seating Zone: <b style={{ color: '#10B981' }}>{selectedPerson.assigned_zone || 'Unassigned / Freely Moving'}</b>
                </div>
                <div style={{ marginTop: '2px', fontSize: '11px' }}>
                  Observation Span: Frames #{selectedPerson.first_frame}–#{selectedPerson.last_frame}
                </div>
              </div>

              <div style={{ fontWeight: 700, margin: '8px 0 4px 0', color: '#94A3B8', fontSize: '11px' }}>
                ASSOCIATED CANDIDATE EVENTS ({personEvents.length}):
              </div>

              {personEvents.map(pe => (
                <div key={pe.event_id} style={{
                  padding: '8px',
                  background: '#1E293B',
                  borderLeft: pe.priority === 'RED' ? '3px solid #EF4444' : '3px solid #F59E0B',
                  borderRadius: '4px',
                  marginBottom: '4px',
                  fontSize: '11px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <b style={{ color: pe.priority === 'RED' ? '#F87171' : '#FCD34D' }}>{pe.event_type.replace(/_/g, ' ')}</b>
                    <span style={{ fontSize: '9px', color: '#94A3B8' }}>{pe.start_time.toFixed(1)}s</span>
                  </div>
                  <div style={{ fontSize: '10px', color: '#CBD5E1', marginTop: '2px' }}>{pe.description}</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: '24px', textAlign: 'center', color: '#64748B', fontSize: '12px' }}>
              Click any tracked person on the video canvas to inspect their seating zone and recorded candidate events.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
