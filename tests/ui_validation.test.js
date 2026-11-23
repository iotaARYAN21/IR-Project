'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { validateSearchPayload } = require('../ui/search-validation');

const validPayload = { query: '  machine learning ', method: 'proximity', k: '10', candidates: '30' };

test('accepts proximity slop boundaries and normalizes the API payload', () => {
  for (const slop of [0, 20]) {
    const result = validateSearchPayload({ ...validPayload, slop });
    assert.deepEqual(result, {
      value: {
        query: 'machine learning', method: 'proximity', k: 10,
        candidates: 30, rerank: false, slop,
      },
    });
  }
});

test('defaults API slop to 2', () => {
  assert.equal(validateSearchPayload(validPayload).value.slop, 2);
});

test('rejects out of range, fractional, and non numeric slop', () => {
  for (const slop of [-1, 21, 1.5, 'near']) {
    assert.equal(
      validateSearchPayload({ ...validPayload, slop }).error,
      'Proximity distance must be between 0 and 20',
    );
  }
});
