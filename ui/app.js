const form = document.querySelector('#search-form');
const queryInput = document.querySelector('#query');
const methodSelect = document.querySelector('#method');
const kInput = document.querySelector('#k');
const candidatesInput = document.querySelector('#candidates');
const rerankInput = document.querySelector('#rerank');
const diversityInput = document.querySelector('#diversity');
const compareInput = document.querySelector('#compare');
const compareMethodField = document.querySelector('#compare-method-field');
const compareMethodSelect = document.querySelector('#compare-method');
const results = document.querySelector('#results');
const status = document.querySelector('#status');
const themeToggle = document.querySelector('#theme-toggle');
const slopInput = document.querySelector('#slop');
const slopField = document.querySelector('#slop-field');
const compareSlopInput = document.querySelector('#compare-slop');
const compareSlopField = document.querySelector('#compare-slop-field');
const proximityHint = document.querySelector('#proximity-hint');
const advancedSearch = document.querySelector('#advanced-search');
const exactPhraseInput = document.querySelector('#exact-phrase');
const titleOnlyInput = document.querySelector('#title-only');
const mustIncludeInput = document.querySelector('#must-include');
const mustExcludeInput = document.querySelector('#must-exclude');
const methods = ['bm25', 'dense', 'hybrid'];
const proximityMethods = ['bm25', 'hybrid'];

function updateCompareMethods() {
  const previous = compareMethodSelect.value;
  const available = methods.filter((method) => method !== methodSelect.value);
  compareMethodSelect.replaceChildren(...available.map((method) => {
    const option = document.createElement('option');
    option.value = method;
    option.textContent = method.toUpperCase();
    return option;
  }));
  compareMethodSelect.value = available.includes(previous)
    ? previous
    : available[0];
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function highlight(text, query) {
  let output = escapeHtml(text);
  const words = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])]
    .filter((word) => word.length > 2)
    .sort((a, b) => b.length - a.length);

  for (const word of words) {
    const pattern = new RegExp(`(${word.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')})`, 'gi');
    output = output.replace(pattern, '<mark>$1</mark>');
  }
  return output;
}

function formatTiming(response) {
  if (!response.timings_ms || !response.timings_ms.total) return '';
  return `${response.timings_ms.total.toFixed(1)} ms total`;
}

function updateSlopVisibility() {
  // The selectors stay in the layout (invisible) so nothing else moves when the
  // method changes.
  const mainOn = proximityMethods.includes(methodSelect.value);
  const compareOn = proximityMethods.includes(compareMethodSelect.value);
  slopField.classList.toggle('is-inactive', !mainOn);
  slopInput.disabled = !mainOn;
  compareSlopField.classList.toggle('is-inactive', !compareOn);
  compareSlopInput.disabled = !compareOn;
  proximityHint.hidden = !(mainOn || (compareInput.checked && compareOn));
  // Custom BM25 search applies to whichever column uses BM25.
  const compared = compareInput.checked ? compareMethodSelect.value : null;
  advancedSearch.hidden = ![methodSelect.value, compared].includes('bm25');
}

function renderColumn(response, query) {
  const cards = response.hits.length
    ? response.hits.map((hit, index) => `
      <article class="result-card">
        <div class="result-top">
          <span class="rank">${index + 1}</span>
          <h3>${escapeHtml(hit.title || 'Untitled document')}</h3>
        </div>
        <p>${highlight(hit.text, query)}</p>
        <div class="score">${response.reranked ? 'CrossEncoder' : response.method === 'dense' ? 'Cosine' : response.method === 'hybrid' ? 'RRF' : 'BM25'}: ${Number(hit.score).toFixed(4)}</div>
      </article>`).join('')
    : '<div class="empty">No results.</div>';

  return `
    <section class="result-column">
      <header class="column-header">
        <h2>${escapeHtml(response.method)}${response.reranked ? ' + rerank' : ''}</h2>
        <span class="timing">${formatTiming(response)}</span>
      </header>
      ${cards}
    </section>`;
}

async function searchOne(query, method, slopInputEl) {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      method,
      k: Number(kInput.value),
      candidates: Number(candidatesInput.value),
      rerank: rerankInput.checked,
      diversity: diversityInput.value !== '' ? Number(diversityInput.value) : null,
      slop: proximityMethods.includes(method) && slopInputEl.value !== ''
        ? Number(slopInputEl.value)
        : null,
      exactPhrase: method === 'bm25' && exactPhraseInput.checked,
      titleOnly: method === 'bm25' && titleOnlyInput.checked,
      mustInclude: method === 'bm25' ? mustIncludeInput.value : '',
      mustExclude: method === 'bm25' ? mustExcludeInput.value : ''
    })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Search failed');
  return data;
}

form.addEventListener('submit', async (event) => {
  event.preventEvent?.();
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) return;

  status.className = 'status';
  status.textContent = 'Searching…';
  results.innerHTML = '';

  try {
    const selected = methodSelect.value;
    const searches = [[selected, slopInput]];
    if (compareInput.checked) searches.push([compareMethodSelect.value, compareSlopInput]);

    const responses = await Promise.all(
      searches.map(([method, slopInputEl]) => searchOne(query, method, slopInputEl))
    );
    results.style.setProperty('--columns', String(responses.length));
    results.innerHTML = responses.map((response) => renderColumn(response, query)).join('');

    const stageTimings = responses.flatMap((response) =>
      Object.entries(response.timings_ms || {})
        .filter(([name]) => name !== 'total')
        .map(([name, value]) => `${name} ${value.toFixed(1)} ms`)
    );
    status.textContent = stageTimings.length
      ? `${responses.length} method${responses.length === 1 ? '' : 's'} · ${stageTimings.join(' · ')}`
      : `${responses.length} result${responses.length === 1 ? '' : 's'}`;
  } catch (error) {
    status.className = 'status error';
    status.textContent = error.message;
  }
});

methodSelect.addEventListener('change', () => {
  updateCompareMethods();
  updateSlopVisibility();
});
compareInput.addEventListener('change', () => {
  compareMethodField.hidden = !compareInput.checked;
  updateSlopVisibility();
});
compareMethodSelect.addEventListener('change', updateSlopVisibility);
updateCompareMethods();
updateSlopVisibility();

const savedTheme = localStorage.getItem('ir-theme');
if (savedTheme) {
  document.documentElement.dataset.theme = savedTheme;
  themeToggle.textContent = savedTheme === 'dark' ? 'Light' : 'Dark';
}

themeToggle.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  themeToggle.textContent = dark ? 'Light' : 'Dark';
  localStorage.setItem('ir-theme', dark ? 'dark' : 'light');
});
