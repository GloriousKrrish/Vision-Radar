import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { apiUrl } from '../config/api';

export interface ProcessingJobRecord {
  id: number;
  video_id: number;
  calibration_id?: number | null;
  status: 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELLED';
  stage: string;
  progress_pct: number;
  error_message?: string | null;
  telemetry?: any;
  config_json?: any;
  domain?: 'traffic' | 'examination';
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface SharedJobContextType {
  activeJobId: number | null;
  activeJob: ProcessingJobRecord | null;
  activeDomain: 'traffic' | 'examination';
  setDomain: (domain: 'traffic' | 'examination', preserveJobId?: number) => void;
  isLoadingJob: boolean;
  jobList: ProcessingJobRecord[];
  selectJob: (jobId: number) => Promise<void>;
  refreshJobs: () => Promise<void>;
  exportModalOpen: boolean;
  exportTargetJobId: number | null;
  openExportModal: (jobId?: number) => void;
  closeExportModal: () => void;
}

const SharedJobContext = createContext<SharedJobContextType | undefined>(undefined);

export const SharedJobProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeJobId, setActiveJobId] = useState<number | null>(() => {
    try {
      const saved = localStorage.getItem('visionradar_active_job_id');
      return saved ? parseInt(saved, 10) : null;
    } catch {
      return null;
    }
  });

  const [activeJob, setActiveJob] = useState<ProcessingJobRecord | null>(null);
  const [activeDomain, setActiveDomain] = useState<'traffic' | 'examination'>(() => {
    try {
      const saved = localStorage.getItem('visionradar_active_domain');
      if (saved === 'examination' || saved === 'traffic') return saved;
      return 'traffic';
    } catch {
      return 'traffic';
    }
  });
  const [isLoadingJob, setIsLoadingJob] = useState<boolean>(false);
  const [jobList, setJobList] = useState<ProcessingJobRecord[]>([]);

  const [exportModalOpen, setExportModalOpen] = useState<boolean>(false);
  const [exportTargetJobId, setExportTargetJobId] = useState<number | null>(null);

  // Refresh available jobs list
  const refreshJobs = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/api/v1/jobs'));
      if (res.ok) {
        const data: ProcessingJobRecord[] = await res.json();
        setJobList(data);

        // If no active job selected yet, auto-select latest matching activeDomain
        if (!activeJobId && data.length > 0) {
          const matching = data.find(j => {
            const d = (j.domain || ((j.config_json || {}).domain || (j.config_json || {}).module || '').toLowerCase());
            return activeDomain === 'examination' ? d.includes('exam') : !d.includes('exam');
          });
          if (matching) {
            selectJob(matching.id);
          }
        }
      }
    } catch (err) {
      console.warn('[SharedJobContext] Could not fetch jobs list:', err);
    }
  }, [activeJobId]);

  // Select and load specific job
  const selectJob = useCallback(async (jobId: number) => {
    setIsLoadingJob(true);
    try {
      const res = await fetch(apiUrl(`/api/v1/jobs/${jobId}`));
      if (res.ok) {
        const job: ProcessingJobRecord = await res.json();
        setActiveJob(job);
        setActiveJobId(job.id);
        try {
          localStorage.setItem('visionradar_active_job_id', String(job.id));
        } catch {
          // ignore localStorage error
        }

        const authDomain = job.domain || ((job.config_json || {}).domain || (job.config_json || {}).module || '').toLowerCase();
        setActiveDomain(authDomain.includes('exam') ? 'examination' : 'traffic');
      } else {
        console.warn(`[SharedJobContext] Job #${jobId} not found`);
      }
    } catch (err) {
      console.error(`[SharedJobContext] Error fetching job #${jobId}:`, err);
    } finally {
      setIsLoadingJob(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    refreshJobs();
  }, [refreshJobs]);

  // Load active job when activeJobId changes
  useEffect(() => {
    if (activeJobId && (!activeJob || activeJob.id !== activeJobId)) {
      selectJob(activeJobId);
    }
  }, [activeJobId, activeJob, selectJob]);

  // Live polling if active job is in progress
  useEffect(() => {
    if (!activeJob || ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(activeJob.status)) {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch(apiUrl(`/api/v1/jobs/${activeJob.id}`));
        if (res.ok) {
          const updated: ProcessingJobRecord = await res.json();
          setActiveJob(updated);
        }
      } catch {
        // silent polling catch
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeJob]);

  // Change authoritative domain without mutating any existing running jobs
  const setDomain = useCallback(async (targetDomain: 'traffic' | 'examination', preserveJobId?: number) => {
    setActiveDomain(targetDomain);
    try {
      localStorage.setItem('visionradar_active_domain', targetDomain);
    } catch {
      // ignore localStorage error
    }

    if (preserveJobId) {
      if (activeJobId !== preserveJobId) {
        selectJob(preserveJobId);
      }
      return;
    }

    // Determine current active job domain
    const currentJobDomain = activeJob
      ? (activeJob.domain || ((activeJob.config_json || {}).domain || (activeJob.config_json || {}).module || '').toLowerCase().includes('exam') ? 'examination' : 'traffic')
      : null;

    if (currentJobDomain === targetDomain && activeJobId !== null) {
      return; // Already aligned with target domain
    }

    // Look for matching job in candidate list or fetch freshly
    let candidateList = jobList;
    if (!candidateList || candidateList.length === 0) {
      try {
        const res = await fetch(apiUrl('/api/v1/jobs?status=SUCCEEDED&limit=30'));
        if (res.ok) {
          candidateList = await res.json();
          setJobList(candidateList);
        }
      } catch {}
    }

    const matchingJob = candidateList.find(j => {
      const d = (j.domain || ((j.config_json || {}).domain || (j.config_json || {}).module || '').toLowerCase());
      return targetDomain === 'examination' ? d.includes('exam') : !d.includes('exam');
    });

    if (matchingJob) {
      selectJob(matchingJob.id);
    } else {
      setActiveJobId(null);
      setActiveJob(null);
      try {
        localStorage.removeItem('visionradar_active_job_id');
      } catch {}
    }
  }, [activeJob, activeJobId, jobList, selectJob]);

  const openExportModal = (jobId?: number) => {
    setExportTargetJobId(jobId || activeJobId);
    setExportModalOpen(true);
  };

  const closeExportModal = () => {
    setExportModalOpen(false);
  };

  return (
    <SharedJobContext.Provider
      value={{
        activeJobId,
        activeJob,
        activeDomain,
        setDomain,
        isLoadingJob,
        jobList,
        selectJob,
        refreshJobs,
        exportModalOpen,
        exportTargetJobId,
        openExportModal,
        closeExportModal,
      }}
    >
      {children}
    </SharedJobContext.Provider>
  );
};

export const useSharedJob = (): SharedJobContextType => {
  const context = useContext(SharedJobContext);
  if (!context) {
    throw new Error('useSharedJob must be used within a SharedJobProvider');
  }
  return context;
};
