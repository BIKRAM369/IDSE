/**
 * api.js — Thin fetch wrappers for the Flask backend
 */

const API = (() => {
  const BASE = '';  // served from same origin

  async function post(endpoint) {
    const res = await fetch(BASE + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    const json = await res.json();
    if (!json.success) throw new Error(json.error || 'Unknown error');
    return json.data;
  }

  async function get(endpoint) {
    const res = await fetch(BASE + endpoint);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  return { post, get };
})();
