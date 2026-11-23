const form = document.querySelector('#search-form');
const queryInput = document.querySelector('#query');
const methodSelect = document.querySelector('#method');
const kSelect = document.querySelector('#k');
const candidatesSelect = document.querySelector('#candidates');
const rerankInput = document.querySelector('#rerank');
const compareInput = document.querySelector('#compare');
const results = document.querySelector('#results');
const status = document.querySelector('#status');
const themeToggle = document.querySelector('#theme-toggle');

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

async function searchOne(query, method) {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      method,
      k: Number(kSelect.value),
      candidates: Number(candidatesSelect.value),
      rerank: rerankInput.checked
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
    const methods = compareInput.checked
      ? [selected === 'hybrid' ? 'bm25' : selected, selected]
      : [selected];

    const uniqueMethods = [...new Set(methods)];
    const responses = await Promise.all(uniqueMethods.map((method) => searchOne(query, method)));
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
