'use strict';

function validateSearchPayload(payload) {
  if (!payload.query || typeof payload.query !== 'string') {
    return { error: 'A query is required' };
  }
  if (!['bm25', 'dense', 'hybrid', 'proximity'].includes(payload.method)) {
    return { error: 'Invalid search method' };
  }
  const k = Number(payload.k);
  const candidates = Number(payload.candidates);
  if (!Number.isSafeInteger(k) || k < 1 || k > 1000) {
    return { error: 'Results must be a whole number from 1 to 1000' };
  }
  if (!Number.isSafeInteger(candidates) || candidates < 1 || candidates > 1000) {
    return { error: 'Candidates must be a whole number from 1 to 1000' };
  }
  const slop = Number(payload.slop ?? 2);
  if (!Number.isSafeInteger(slop) || slop < 0 || slop > 20) {
    return { error: 'Proximity distance must be between 0 and 20' };
  }
  return {
    value: {
      query: payload.query.trim(),
      method: payload.method,
      k,
      candidates,
      rerank: Boolean(payload.rerank),
      slop,
    },
  };
}

module.exports = { validateSearchPayload };
