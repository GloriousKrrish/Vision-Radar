/**
 * VisionRadar API & WebSocket Runtime Configuration
 *
 * Automatically resolves endpoint targets:
 * - When VITE_API_URL is configured, requests and WebSockets route to that target.
 * - When running on Vercel (*.vercel.app), automatically targets Render backend:
 *   https://visionradar-backend.onrender.com (and wss:// for WebSockets).
 * - Allows runtime override via localStorage key 'visionradar_api_url'.
 * - In local development, remains relative to window.location (port 8080 -> proxy to 8000).
 */

const metaEnv: Record<string, string | undefined> = (typeof import.meta !== 'undefined' && (import.meta as any).env) || {};

function resolveBaseUrl(): string {
  // 1. Explicit Vite env variable
  if (metaEnv.VITE_API_URL) {
    return metaEnv.VITE_API_URL.replace(/\/$/, '');
  }

  if (typeof window !== 'undefined') {
    // 2. Local storage override for custom proctor instances
    const stored = localStorage.getItem('visionradar_api_url');
    if (stored) {
      return stored.replace(/\/$/, '');
    }

    // 3. Automatic Vercel production deployment fallback to Render backend
    if (window.location.hostname.endsWith('vercel.app')) {
      return 'https://visionradar-backend.onrender.com';
    }
  }

  // 4. Localhost / same-origin relative fallback
  return '';
}

export const API_BASE_URL: string = resolveBaseUrl();

export function apiUrl(path: string): string {
  if (!API_BASE_URL) return path;
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

export function getWsUrl(path: string): string {
  const explicitWs = metaEnv.VITE_WS_URL?.replace(/\/$/, '');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;

  if (explicitWs) {
    return `${explicitWs}${cleanPath}`;
  }

  if (API_BASE_URL.startsWith('http')) {
    const wsProto = API_BASE_URL.startsWith('https') ? 'wss:' : 'ws:';
    const host = API_BASE_URL.replace(/^https?:\/\//, '');
    return `${wsProto}//${host}${cleanPath}`;
  }

  const proto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws';
  const host = typeof window !== 'undefined' ? window.location.host : 'localhost:8000';
  return `${proto}://${host}${cleanPath}`;
}
