/**
 * VisionRadar API & WebSocket Runtime Configuration
 *
 * Automatically resolves endpoint targets:
 * - When VITE_API_URL is configured (e.g., Vercel -> Render backend), requests and
 *   WebSockets route to the deployed Render service.
 * - In local development or unified host mode, paths remain relative to window.location.
 */

const metaEnv: Record<string, string | undefined> = (typeof import.meta !== 'undefined' && (import.meta as any).env) || {};

export const API_BASE_URL: string = (metaEnv.VITE_API_URL || '').replace(/\/$/, '');

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
