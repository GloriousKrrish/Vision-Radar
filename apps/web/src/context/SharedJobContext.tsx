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
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface SharedJobContextType {
  activeJobId: number | null;
  activeJob: ProcessingJobRecord | null;
  activeDomain: 'traffic' | 'examination';
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
  const [activeDomain, setActiveDomain] = useState<'traffic' | 'examination'>('traffic');
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

        // If no active job selected yet, auto-select latest
        if (!activeJobId && data.length > 0) {
          selectJob(data[0].id);
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

        const cfg = job.config_json || {};
        const mod = (cfg.module || '').toLowerCase();
        setActiveDomain(mod.includes('exam') ? 'examination' : 'traffic');
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
