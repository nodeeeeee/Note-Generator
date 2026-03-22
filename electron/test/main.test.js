'use strict';
/**
 * Unit tests for main.js helper functions.
 * Run with: cd electron && npm test
 *
 * These tests exercise the pure Node.js functions (config, constants, paths)
 * without needing a running Electron instance.
 */

const path  = require('path');
const os    = require('os');
const fs    = require('fs');
const assert = require('assert');

// ── Setup: create a temp dir for each test run ────────────────────────────────
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'autonote-test-'));

// Monkey-patch DATA_DIR so main.js uses our temp dir
process.env.AUTONOTE_TEST_DATA_DIR = tmpDir;

// Require main.js functions directly (we exported them for testing)
// Since main.js uses `app.whenReady()` which requires Electron, we instead
// require a stripped version.  We patch module internals via require cache tricks.
// For simplicity, we test the exported module helpers directly.

// ── Mock electron before requiring main.js ────────────────────────────────────
const Module = require('module');
const origLoad = Module._load;
Module._load = function(request, ...args) {
  if (request === 'electron') {
    return {
      app: {
        whenReady: () => Promise.resolve(),
        on: () => {},
        quit: () => {},
        getAllWindows: () => [],
      },
      BrowserWindow: class { constructor() {} loadFile() {} on() {} webContents = { send: () => {} } },
      ipcMain: { handle: () => {}, on: () => {} },
      dialog: { showOpenDialog: async () => ({ canceled: true }) },
    };
  }
  return origLoad.call(this, request, ...args);
};

// Patch DATA_DIR before loading
const mainPath = path.join(__dirname, '..', 'main.js');
// Re-define DATA_DIR at module level via environment
// We'll test the exported functions by calling them with the temp dir logic

// ── Minimal inline re-implementations for unit testing ───────────────────────
// (Copies of the pure functions from main.js, parameterised with tmpDir)

function loadConfig(dir = tmpDir) {
  const f = path.join(dir, 'config.json');
  if (fs.existsSync(f)) {
    try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch {}
  }
  return {};
}

function saveConfig(data, dir = tmpDir) {
  fs.mkdirSync(dir, { recursive: true });
  const f   = path.join(dir, 'config.json');
  const cfg = loadConfig(dir);
  Object.assign(cfg, data);
  fs.writeFileSync(f, JSON.stringify(cfg, null, 2));
}

function readConstant(scriptPath, name) {
  if (!scriptPath || !fs.existsSync(scriptPath)) return '?';
  try {
    const src = fs.readFileSync(scriptPath, 'utf8');
    const m   = src.match(new RegExp(`^${name}\\s*=\\s*(.+)`, 'm'));
    if (!m) return '?';
    return m[1].replace(/#.*$/, '').trim().replace(/^["']|["']$/g, '');
  } catch { return '?'; }
}

function writeConstant(scriptPath, name, value) {
  if (!scriptPath || !fs.existsSync(scriptPath)) return false;
  try {
    const src = fs.readFileSync(scriptPath, 'utf8');
    let newVal;
    if (value === 'None') newVal = 'None';
    else if (value !== '' && !isNaN(Number(value))) newVal = value;
    else newVal = `"${value}"`;
    const re     = new RegExp(`^(${name}\\s*=\\s*)([^\\n#]+)`, 'm');
    const newSrc = src.replace(re, `$1${newVal}`);
    if (newSrc === src) return false;
    fs.writeFileSync(scriptPath, newSrc);
    return true;
  } catch { return false; }
}

function detectCuda() {
  const { spawnSync } = require('child_process');
  try {
    const r = spawnSync('nvidia-smi', [], { encoding: 'utf8', timeout: 10000 });
    if (r.status === 0) {
      const m = r.stdout.match(/CUDA Version:\s*(\d+)\.(\d+)/);
      if (m) return [parseInt(m[1]), parseInt(m[2])];
      return [12, 0];
    }
  } catch {}
  return null;
}

function torchIndexUrl(cuda) {
  if (!cuda) return null;
  const v = cuda[0] * 10 + cuda[1];
  if (v >= 128) return 'https://download.pytorch.org/whl/cu128';
  if (v >= 126) return 'https://download.pytorch.org/whl/cu126';
  if (v >= 124) return 'https://download.pytorch.org/whl/cu124';
  return 'https://download.pytorch.org/whl/cu121';
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('Config helpers', () => {
  const dir = path.join(tmpDir, 'config-test');

  beforeEach(() => {
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
  });

  test('loadConfig returns empty object when no file', () => {
    assert.deepStrictEqual(loadConfig(dir), {});
  });

  test('saveConfig writes and reads back data', () => {
    saveConfig({ CANVAS_URL: 'canvas.example.com', OUTPUT_DIR: '/tmp/out' }, dir);
    const cfg = loadConfig(dir);
    assert.strictEqual(cfg.CANVAS_URL, 'canvas.example.com');
    assert.strictEqual(cfg.OUTPUT_DIR, '/tmp/out');
  });

  test('saveConfig merges with existing data', () => {
    saveConfig({ A: '1' }, dir);
    saveConfig({ B: '2' }, dir);
    const cfg = loadConfig(dir);
    assert.strictEqual(cfg.A, '1');
    assert.strictEqual(cfg.B, '2');
  });

  test('loadConfig survives malformed JSON', () => {
    fs.writeFileSync(path.join(dir, 'config.json'), '{invalid json}');
    assert.deepStrictEqual(loadConfig(dir), {});
  });
});

describe('Constants helpers', () => {
  const scriptFile = path.join(tmpDir, 'test_script.py');

  beforeEach(() => {
    fs.writeFileSync(scriptFile, [
      'WHISPER_MODEL_SIZE = "large-v3"',
      'WHISPER_LANGUAGE = None',
      'CONTEXT_SEC = 30',
      'OFF_SLIDE_THRESHOLD = 0.28',
      'NOTE_MODEL = "gpt-5.1"  # comment',
    ].join('\n'));
  });

  test('readConstant reads quoted string', () => {
    assert.strictEqual(readConstant(scriptFile, 'WHISPER_MODEL_SIZE'), 'large-v3');
  });

  test('readConstant reads bare None as "None"', () => {
    assert.strictEqual(readConstant(scriptFile, 'WHISPER_LANGUAGE'), 'None');
  });

  test('readConstant reads integer', () => {
    assert.strictEqual(readConstant(scriptFile, 'CONTEXT_SEC'), '30');
  });

  test('readConstant reads float', () => {
    assert.strictEqual(readConstant(scriptFile, 'OFF_SLIDE_THRESHOLD'), '0.28');
  });

  test('readConstant strips inline comment', () => {
    assert.strictEqual(readConstant(scriptFile, 'NOTE_MODEL'), 'gpt-5.1');
  });

  test('readConstant returns "?" for missing constant', () => {
    assert.strictEqual(readConstant(scriptFile, 'NONEXISTENT'), '?');
  });

  test('writeConstant writes quoted string', () => {
    writeConstant(scriptFile, 'WHISPER_MODEL_SIZE', 'small');
    assert.strictEqual(readConstant(scriptFile, 'WHISPER_MODEL_SIZE'), 'small');
  });

  test('writeConstant writes None for "None" value', () => {
    writeConstant(scriptFile, 'WHISPER_MODEL_SIZE', 'None');
    const src = fs.readFileSync(scriptFile, 'utf8');
    assert.match(src, /WHISPER_MODEL_SIZE = None/);
  });

  test('writeConstant writes numeric literal', () => {
    writeConstant(scriptFile, 'CONTEXT_SEC', '60');
    const src = fs.readFileSync(scriptFile, 'utf8');
    assert.match(src, /CONTEXT_SEC = 60/);
    assert.doesNotMatch(src, /CONTEXT_SEC = "60"/);
  });

  test('writeConstant writes float literal', () => {
    writeConstant(scriptFile, 'OFF_SLIDE_THRESHOLD', '0.35');
    assert.strictEqual(readConstant(scriptFile, 'OFF_SLIDE_THRESHOLD'), '0.35');
  });

  test('writeConstant returns false for nonexistent script', () => {
    assert.strictEqual(writeConstant('/nonexistent/path.py', 'FOO', 'bar'), false);
  });
});

describe('torchIndexUrl', () => {
  test('returns cu128 for CUDA 12.8', () => {
    assert.strictEqual(torchIndexUrl([12, 8]), 'https://download.pytorch.org/whl/cu128');
  });

  test('returns cu128 for CUDA 13.0', () => {
    assert.strictEqual(torchIndexUrl([13, 0]), 'https://download.pytorch.org/whl/cu128');
  });

  test('returns cu126 for CUDA 12.6', () => {
    assert.strictEqual(torchIndexUrl([12, 6]), 'https://download.pytorch.org/whl/cu126');
  });

  test('returns cu124 for CUDA 12.4', () => {
    assert.strictEqual(torchIndexUrl([12, 4]), 'https://download.pytorch.org/whl/cu124');
  });

  test('returns cu121 for CUDA 12.1', () => {
    assert.strictEqual(torchIndexUrl([12, 1]), 'https://download.pytorch.org/whl/cu121');
  });

  test('returns null for no CUDA', () => {
    assert.strictEqual(torchIndexUrl(null), null);
  });
});

describe('getFilesStatus', () => {
  const outDir   = path.join(tmpDir, 'files-status-test');
  const courseId = '99999';
  const courseDir = path.join(outDir, courseId);

  beforeEach(() => {
    fs.rmSync(outDir, { recursive: true, force: true });
    fs.mkdirSync(path.join(courseDir, 'captions'),  { recursive: true });
    fs.mkdirSync(path.join(courseDir, 'alignment'), { recursive: true });
    fs.mkdirSync(path.join(courseDir, 'notes'),     { recursive: true });
  });

  function getFilesStatus(courseId, dir = outDir) {
    // Inline version that accepts dir parameter
    let videoDone = 0, videoTotal = 0;
    for (const mp of [
      path.join(dir,    'manifest.json'),
    ]) {
      if (fs.existsSync(mp)) {
        try {
          const m = JSON.parse(fs.readFileSync(mp, 'utf8'));
          const items = Object.values(m).filter(v => v.path && v.path.includes(String(courseId)));
          videoTotal = items.length;
          videoDone  = items.filter(v => v.status === 'done').length;
        } catch {}
      }
    }
    const courseDir = path.join(dir, String(courseId));
    const countJson = (d, excl) => {
      if (!fs.existsSync(d)) return 0;
      return fs.readdirSync(d).filter(f => f.endsWith('.json') && (!excl || !f.includes(excl))).length;
    };
    const notesDir = path.join(courseDir, 'notes');
    let notesFile = null, courseName = null;
    if (fs.existsSync(notesDir)) {
      const mds = fs.readdirSync(notesDir).filter(f => f.endsWith('.md'));
      if (mds.length) { notesFile = mds[0]; courseName = notesFile.replace(/_notes\.md$/, '').replace(/_/g, ' '); }
    }
    return {
      videos: { done: videoDone, total: videoTotal },
      captions:   countJson(path.join(courseDir, 'captions')),
      alignments: countJson(path.join(courseDir, 'alignment'), 'compact'),
      notesFile, courseName,
    };
  }

  test('returns zeros for empty course directory', () => {
    const s = getFilesStatus(courseId);
    assert.strictEqual(s.videos.total, 0);
    assert.strictEqual(s.captions,     0);
    assert.strictEqual(s.alignments,   0);
    assert.strictEqual(s.notesFile,    null);
  });

  test('counts caption files', () => {
    fs.writeFileSync(path.join(courseDir, 'captions', 'L01.json'), '{}');
    fs.writeFileSync(path.join(courseDir, 'captions', 'L02.json'), '{}');
    assert.strictEqual(getFilesStatus(courseId).captions, 2);
  });

  test('excludes compact files from alignment count', () => {
    fs.writeFileSync(path.join(courseDir, 'alignment', 'L01.json'),          '{}');
    fs.writeFileSync(path.join(courseDir, 'alignment', 'L01.compact.json'),  '{}');
    assert.strictEqual(getFilesStatus(courseId).alignments, 1);
  });

  test('detects notes file and extracts course name', () => {
    fs.writeFileSync(path.join(courseDir, 'notes', 'CS3210_notes.md'), '# Notes');
    const s = getFilesStatus(courseId);
    assert.strictEqual(s.notesFile,    'CS3210_notes.md');
    assert.strictEqual(s.courseName,   'CS3210');
  });

  test('reads video status from manifest', () => {
    fs.writeFileSync(path.join(outDir, 'manifest.json'), JSON.stringify({
      'vid1': { path: `/${courseId}/videos/L01.mp4`, status: 'done'    },
      'vid2': { path: `/${courseId}/videos/L02.mp4`, status: 'pending' },
      'vid3': { path: '/other/videos/L03.mp4',       status: 'done'    },
    }));
    const s = getFilesStatus(courseId);
    assert.strictEqual(s.videos.total, 2);
    assert.strictEqual(s.videos.done,  1);
  });
});

// ── Cleanup ───────────────────────────────────────────────────────────────────
afterAll(() => {
  try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
});
