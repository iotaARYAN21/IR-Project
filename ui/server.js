'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { validateSearchPayload, validateSuggestQuery } = require('./search-validation');

const host = process.env.UI_HOST || '127.0.0.1';
const port = Number(process.env.UI_PORT || 3000);
const uiRoot = __dirname;

const pythonCode = String.raw`
import json
import sys
from argparse import Namespace

stdout = sys.stdout
sys.stdout = sys.stderr

from src.search.cli import build_engine

args = Namespace(
    embeddings_dir=None,
    index=${JSON.stringify(process.env.ELASTICSEARCH_INDEX || 'nq_documents')},
    depth=100,
    rrf_k=60,
    bm25_weight=1.0,
    dense_weight=1.0,
)
engine = build_engine(args)

original_print = print
while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        request = json.loads(line)
        if request.get("action") == "suggest":
            titles = engine.suggest(request["prefix"], request.get("limit", 8))
            original_print(json.dumps({"suggestions": titles}), file=stdout, flush=True)
            continue
        response = engine.search(
            request["query"],
            method=request["method"],
            k=request["k"],
            rerank=request["rerank"],
            candidates=request["candidates"],
            slop=request.get("slop"),
            diversity=request.get("diversity"),
            title_only=request.get("titleOnly", False),
            exact_phrase=request.get("exactPhrase", False),
            must_include=request.get("mustInclude") or None,
            must_exclude=request.get("mustExclude") or None,
        )
        payload = {
            "query": response.query,
            "method": response.method,
            "reranked": response.reranked,
            "hits": [
                {"doc_id": h.doc_id, "title": h.title, "text": h.text, "score": h.score}
                for h in response.hits
            ],
            "timings_ms": response.timings_ms,
        }
        original_print(json.dumps(payload), file=stdout, flush=True)
    except Exception as exc:
        original_print(json.dumps({"error": str(exc)}), file=stdout, flush=True)
`;

let bridge;
let bridgeQueue = [];
let restarting = false;

function startBridge() {
  const venvPython = path.resolve(uiRoot, '../.venv/bin/python3');
  const pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python3';
  bridge = spawn(pythonExecutable, ['-u', '-c', pythonCode], {
    cwd: path.resolve(uiRoot, '..'),
    env: process.env,
    stdio: ['pipe', 'pipe', 'inherit']
  });

  bridge.stdout.setEncoding('utf8');
  let buffer = '';

  bridge.stdout.on('data', (chunk) => {
    buffer += chunk;
    let newline;
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      const waiter = bridgeQueue.shift();
      if (waiter) {
        try { waiter(JSON.parse(line)); }
        catch (error) { waiter(null, error); }
      }
    }
  });

  bridge.on('exit', (code) => {
    console.error(`Python search bridge exited (${code ?? 'unknown'}).`);
    for (const waiter of bridgeQueue) waiter(null, new Error('Search service stopped')); 
    bridgeQueue = [];
    if (!restarting) {
      restarting = true;
      setTimeout(() => { restarting = false; startBridge(); }, 1000);
    }
  });
}

function sendToBridge(payload) {
  return new Promise((resolve, reject) => {
    if (!bridge || bridge.exitCode !== null) {
      reject(new Error('Search service is starting; try again'));
      return;
    }
    bridgeQueue.push((value, error) => error ? reject(error) : resolve(value));
    bridge.stdin.write(`${JSON.stringify(payload)}\n`);
  });
}

function sendJson(response, status, value) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body)
  });
  response.end(body);
}

function contentType(filePath) {
  switch (path.extname(filePath)) {
    case '.html': return 'text/html; charset=utf-8';
    case '.css': return 'text/css; charset=utf-8';
    case '.js': return 'text/javascript; charset=utf-8';
    default: return 'application/octet-stream';
  }
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);

  if (request.method === 'GET' && url.pathname === '/api/status') {
    sendJson(response, 200, { status: bridge && bridge.exitCode === null ? 'ready' : 'starting' });
    return;
  }

  if (request.method === 'GET' && url.pathname === '/api/suggest') {
    const validation = validateSuggestQuery(url.searchParams.get('q') || '');
    if (validation.error) {
      sendJson(response, 400, { error: validation.error });
      return;
    }
    if (validation.tooShort) {
      sendJson(response, 200, { suggestions: [] });
      return;
    }
    try {
      const result = await sendToBridge({ action: 'suggest', prefix: validation.value, limit: 8 });
      if (result && result.error) sendJson(response, 500, { error: result.error });
      else sendJson(response, 200, result);
    } catch (error) {
      sendJson(response, 500, { error: error.message || 'Suggestions failed' });
    }
    return;
  }

  if (request.method === 'POST' && url.pathname === '/api/search') {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
      if (body.length > 10000) request.destroy();
    });
    request.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        const validation = validateSearchPayload(payload);
        if (validation.error) {
          sendJson(response, 400, { error: validation.error });
          return;
        }
        const result = await sendToBridge(validation.value);
        if (result && result.error) {
          sendJson(response, 500, { error: result.error });
        } else {
          sendJson(response, 200, result);
        }
      } catch (error) {
        sendJson(response, 500, { error: error.message || 'Search failed' });
      }
    });
    return;
  }

  if (request.method !== 'GET') {
    sendJson(response, 405, { error: 'Method not allowed' });
    return;
  }

  const requested = url.pathname === '/' ? '/index.html' : url.pathname;
  const filePath = path.resolve(uiRoot, `.${requested}`);
  if (!filePath.startsWith(uiRoot + path.sep)) {
    sendJson(response, 403, { error: 'Forbidden' });
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      sendJson(response, 404, { error: 'Not found' });
      return;
    }
    response.writeHead(200, { 'Content-Type': contentType(filePath) });
    response.end(data);
  });
});

startBridge();
server.listen(port, host, () => {
  console.log(`IR Search UI: http://${host}:${port}`);
});
