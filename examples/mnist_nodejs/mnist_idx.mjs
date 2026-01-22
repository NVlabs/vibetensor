import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import zlib from 'node:zlib';
import { promisify } from 'node:util';
import os from 'node:os';

const gunzip = promisify(zlib.gunzip);

const MNIST_FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
};

const DEFAULT_MNIST_BASE_URLS = [
  'https://yann.lecun.com/exdb/mnist',
  'https://storage.googleapis.com/cvdf-datasets/mnist',
];

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true });
}

function download(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fsSync.createWriteStream(destPath);
    const req = https.get(url, (res) => {
      if (res.statusCode !== 200) {
        file.close(() => {});
        fsSync.unlink(destPath, () => {});
        reject(new Error(`download failed: ${url} (status ${res.statusCode})`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
    });
    req.on('error', (err) => {
      file.close(() => {});
      fsSync.unlink(destPath, () => {});
      reject(err);
    });
  });
}

async function readGzFile(p) {
  const gz = await fs.readFile(p);
  return gunzip(gz);
}

function readU32BE(buf, off) {
  return (
    (buf[off] << 24) |
    (buf[off + 1] << 16) |
    (buf[off + 2] << 8) |
    buf[off + 3]
  ) >>> 0;
}

function maybeResolveLocalFile(rootDir, gzName) {
  const gzPath = path.join(rootDir, gzName);
  if (fsSync.existsSync(gzPath)) return { kind: 'gz', path: gzPath };
  if (gzName.endsWith('.gz')) {
    const rawName = gzName.slice(0, -3);
    const rawPath = path.join(rootDir, rawName);
    if (fsSync.existsSync(rawPath)) return { kind: 'raw', path: rawPath };
  }
  return null;
}

function parseIdxImages(buf) {
  const magic = readU32BE(buf, 0);
  if (magic !== 2051) throw new Error(`bad IDX image magic: ${magic}`);
  const count = readU32BE(buf, 4);
  const rows = readU32BE(buf, 8);
  const cols = readU32BE(buf, 12);
  const expected = 16 + count * rows * cols;
  if (buf.length !== expected) {
    throw new Error(`bad IDX image length: got ${buf.length}, expected ${expected}`);
  }
  const data = new Uint8Array(buf.buffer, buf.byteOffset + 16, count * rows * cols);
  return { count, rows, cols, data };
}

function parseIdxLabels(buf) {
  const magic = readU32BE(buf, 0);
  if (magic !== 2049) throw new Error(`bad IDX label magic: ${magic}`);
  const count = readU32BE(buf, 4);
  const expected = 8 + count;
  if (buf.length !== expected) {
    throw new Error(`bad IDX label length: got ${buf.length}, expected ${expected}`);
  }
  const data = new Uint8Array(buf.buffer, buf.byteOffset + 8, count);
  return { count, data };
}

function requiredFilesList() {
  const lines = [];
  for (const gz of Object.values(MNIST_FILES)) {
    lines.push(`- ${gz}`);
    if (gz.endsWith('.gz')) lines.push(`- ${gz.slice(0, -3)}`);
  }
  return lines.join('\n');
}

function looksLikeMnistDir(dir) {
  for (const gz of Object.values(MNIST_FILES)) {
    if (!maybeResolveLocalFile(dir, gz)) return false;
  }
  return true;
}

function candidateMnistDirs() {
  const home = os.homedir();
  const out = [];
  out.push(path.join(home, '.cache', 'torchvision', 'datasets', 'MNIST', 'raw'));
  out.push(path.join(home, '.cache', 'torchvision', 'MNIST', 'raw'));
  out.push(path.join(home, '.cache', 'torch', 'datasets', 'MNIST', 'raw'));
  out.push(path.join(home, '.cache', 'mnist'));
  out.push('/datasets/mnist');
  out.push('/dataset/mnist');
  return out;
}

async function ensureMnistFiles(rootDir) {
  await ensureDir(rootDir);

  const missing = [];
  for (const fname of Object.values(MNIST_FILES)) {
    if (!maybeResolveLocalFile(rootDir, fname)) missing.push(fname);
  }
  if (missing.length === 0) return;

  const noDownload =
    process.env.MNIST_NO_DOWNLOAD === '1' ||
    process.env.MNIST_NO_DOWNLOAD === 'true' ||
    process.env.MNIST_NO_DOWNLOAD === 'yes';
  if (noDownload) {
    const hints = candidateMnistDirs().filter((d) => {
      try {
        return looksLikeMnistDir(d);
      } catch {
        return false;
      }
    });
    throw new Error(
      [
        'MNIST files are missing and downloads are disabled (MNIST_NO_DOWNLOAD=1).',
        `MNIST_DIR: ${rootDir}`,
        ...(hints.length > 0
          ? [
              'Found an existing MNIST cache directory candidate:',
              ...hints.map((d) => `- ${d}`),
              'Set MNIST_DIR to one of the above, or copy the files into MNIST_DIR.',
            ]
          : []),
        'Required files:',
        requiredFilesList(),
      ].join('\n'),
    );
  }

  const baseFromEnv = process.env.MNIST_BASE_URL;
  const baseUrls = baseFromEnv && baseFromEnv.length > 0
    ? [baseFromEnv, ...DEFAULT_MNIST_BASE_URLS]
    : DEFAULT_MNIST_BASE_URLS;

  for (const fname of missing) {
    const p = path.join(rootDir, fname);
    let lastErr = null;
    for (const base of baseUrls) {
      const url = `${base}/${fname}`;
      try {
        await download(url, p);
        lastErr = null;
        break;
      } catch (e) {
        lastErr = e;
        try { fsSync.unlinkSync(p); } catch {}
      }
    }
    if (lastErr) {
      throw new Error(
        [
          `Failed to download MNIST file: ${fname}`,
          `MNIST_DIR: ${rootDir}`,
          'Tried base URLs:',
          baseUrls.map((u) => `- ${u}`).join('\n'),
          '',
          'If you are offline, download these files elsewhere and copy them into MNIST_DIR:',
          requiredFilesList(),
          '',
          `Last error: ${String(lastErr?.message ?? lastErr)}`,
        ].join('\n'),
      );
    }
  }
}

export async function loadMnist(opts = {}) {
  const rootDir =
    opts.rootDir ??
    process.env.MNIST_DIR ??
    path.join(process.cwd(), '.cache', 'mnist');

  await ensureMnistFiles(rootDir);

  async function readMaybeGz(name) {
    const resolved = maybeResolveLocalFile(rootDir, name);
    if (!resolved) {
      throw new Error(`missing MNIST file: ${name} (searched in ${rootDir})`);
    }
    if (resolved.kind === 'gz') return readGzFile(resolved.path);
    return fs.readFile(resolved.path);
  }

  const [trainImagesRaw, trainLabelsRaw, testImagesRaw, testLabelsRaw] =
    await Promise.all([
      readMaybeGz(MNIST_FILES.trainImages),
      readMaybeGz(MNIST_FILES.trainLabels),
      readMaybeGz(MNIST_FILES.testImages),
      readMaybeGz(MNIST_FILES.testLabels),
    ]);

  const trainImages = parseIdxImages(trainImagesRaw);
  const trainLabels = parseIdxLabels(trainLabelsRaw);
  const testImages = parseIdxImages(testImagesRaw);
  const testLabels = parseIdxLabels(testLabelsRaw);

  if (trainImages.count !== trainLabels.count) throw new Error('train count mismatch');
  if (testImages.count !== testLabels.count) throw new Error('test count mismatch');
  if (trainImages.rows !== 28 || trainImages.cols !== 28) throw new Error('unexpected train image shape');
  if (testImages.rows !== 28 || testImages.cols !== 28) throw new Error('unexpected test image shape');

  return {
    train: { images: trainImages.data, labels: trainLabels.data, count: trainImages.count },
    test: { images: testImages.data, labels: testLabels.data, count: testImages.count },
    rows: 28,
    cols: 28,
  };
}

