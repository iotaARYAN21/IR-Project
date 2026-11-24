'use strict';

function validateSearchPayload(payload) {
  if (!payload.query || typeof payload.query !== 'string') {
    return { error: 'A query is required' };
  }
  if (!['bm25', 'dense', 'hybrid'].includes(payload.method)) {
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
  // Proximity is optional: a missing, null or blank value leaves it off.
  const slopGiven = payload.slop !== undefined && payload.slop !== null && payload.slop !== '';
  const slop = slopGiven ? Number(payload.slop) : null;
  if (slopGiven && (!Number.isSafeInteger(slop) || slop < 0 || slop > 20)) {
    return { error: 'Proximity distance must be between 0 and 20' };
  }
  for (const field of ['mustInclude', 'mustExclude']) {
    if (payload[field] !== undefined && typeof payload[field] !== 'string') {
      return { error: `${field} must be text` };
    }
    if ((payload[field] || '').length > 200) {
      return { error: `${field} must be 200 characters or fewer` };
    }
  }
  return {
    value: {
      query: payload.query.trim(),
      method: payload.method,
      k,
      candidates,
      rerank: Boolean(payload.rerank),
      // Proximity is a BM25 feature: dense search never uses it.
      slop: ['bm25', 'hybrid'].includes(payload.method) ? slop : null,
      titleOnly: Boolean(payload.titleOnly),
      exactPhrase: Boolean(payload.exactPhrase),
      mustInclude: (payload.mustInclude || '').trim(),
      mustExclude: (payload.mustExclude || '').trim(),
    },
  };
}

module.exports = { validateSearchPayload };
