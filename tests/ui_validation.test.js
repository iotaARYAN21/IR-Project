'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { validateSearchPayload } = require('../ui/search-validation');

const validPayload = { query: '  machine learning ', method: 'bm25', k: '10', candidates: '30' };

test('accepts proximity slop boundaries and normalizes the API payload', () => {
  for (const slop of [0, 20]) {
    const result = validateSearchPayload({ ...validPayload, slop });
    assert.deepEqual(result, {
      value: {
        query: 'machine learning', method: 'bm25', k: 10,
        candidates: 30, rerank: false, slop, titleOnly: false,
        exactPhrase: false, mustInclude: '', mustExclude: '',
      },
    });
  }
});

test('leaves proximity off when slop is missing, null or blank', () => {
  for (const slop of [undefined, null, '']) {
    assert.equal(validateSearchPayload({ ...validPayload, slop }).value.slop, null);
  }
});

test('rejects out of range, fractional, and non numeric slop', () => {
  for (const slop of [-1, 21, 1.5, 'near']) {
    assert.equal(
      validateSearchPayload({ ...validPayload, slop }).error,
      'Proximity distance must be between 0 and 20',
    );
  }
});

test('normalizes advanced search options and rejects oversized terms', () => {
  const result = validateSearchPayload({
    ...validPayload, exactPhrase: true, titleOnly: true,
    mustInclude: ' policy ', mustExclude: 'sports',
  });
  assert.equal(result.value.exactPhrase, true);
  assert.equal(result.value.titleOnly, true);
  assert.equal(result.value.mustInclude, 'policy');
  assert.equal(result.value.mustExclude, 'sports');
  assert.match(validateSearchPayload({ ...validPayload, mustInclude: 'x'.repeat(201) }).error, /200 characters/);
});

test('proximity is a BM25 option, not a search method', () => {
  assert.equal(
    validateSearchPayload({ ...validPayload, method: 'proximity' }).error,
    'Invalid search method',
  );
});

test('drops proximity for dense search, which does not use BM25', () => {
  const dense = validateSearchPayload({ ...validPayload, method: 'dense', slop: 3 });
  assert.equal(dense.value.slop, null);
  for (const method of ['bm25', 'hybrid']) {
    assert.equal(validateSearchPayload({ ...validPayload, method, slop: 3 }).value.slop, 3);
  }
});
