if (!process.env.VBT_NODE_MAX_INFLIGHT_OPS) {
  process.env.VBT_NODE_MAX_INFLIGHT_OPS = '64';
}

import { ops, cuda, scalarBool, scalarInt64 } from 'vibetensor';
import { loadMnist } from './mnist_idx.mjs';

function parseArgs(argv) {
  const out = {
    epochs: 1,
    batchSize: 64,
    lr: 0.1,
    logEvery: 100,
    maxSteps: -1,
  };

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`missing value for ${a}`);
      return argv[++i];
    };
    if (a === '--epochs') out.epochs = Number(next());
    else if (a === '--batch-size') out.batchSize = Number(next());
    else if (a === '--lr') out.lr = Number(next());
    else if (a === '--log-every') out.logEvery = Number(next());
    else if (a === '--max-steps') out.maxSteps = Number(next());
    else if (a === '--help' || a === '-h') {
      out.help = true;
    } else {
      throw new Error(`unknown arg: ${a}`);
    }
  }

  return out;
}

function argMax10(probsB10) {
  let best = 0;
  let bestV = probsB10[0];
  for (let c = 1; c < 10; c++) {
    const v = probsB10[c];
    if (v > bestV) {
      bestV = v;
      best = c;
    }
  }
  return best;
}

function makeBatch(imagesU8, labelsU8, start, batchSize) {
  const imgSize = 28 * 28;
  const n = Math.min(batchSize, Math.floor((imagesU8.length / imgSize) - start));
  const x = new Float32Array(n * 10 * imgSize);
  const y = new Float32Array(n * 10 * 1);
  const yIdx = new Uint8Array(n);

  for (let b = 0; b < n; b++) {
    const label = labelsU8[start + b] | 0;
    yIdx[b] = label;
    y[(b * 10 + label) * 1] = 1.0;

    const imgOff = (start + b) * imgSize;
    for (let c = 0; c < 10; c++) {
      const outOff = (b * 10 + c) * imgSize;
      for (let i = 0; i < imgSize; i++) {
        x[outOff + i] = imagesU8[imgOff + i] / 255.0;
      }
    }
  }

  return { n, x, y, yIdx };
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(
      [
        'Usage: node train_mnist.mjs [--epochs N] [--batch-size N] [--lr LR] [--log-every N] [--max-steps N]',
        '',
        'Notes:',
        '- Requires CUDA (uses vibetensor Node bindings + vt::* ops).',
      ].join('\n'),
    );
    return;
  }

  if (!cuda.isAvailable()) {
    throw new Error('CUDA is not available (vibetensor cuda.isAvailable() == false)');
  }

  cuda.setDevice(0);

  const mnist = await loadMnist();
  const trainImages = mnist.train.images;
  const trainLabels = mnist.train.labels;

  const imgSize = 28 * 28;
  const B = args.batchSize;

  const dim0 = await scalarInt64(0);
  const dim1 = await scalarInt64(1);
  const dim2 = await scalarInt64(2);
  const keepTrue = await scalarBool(true);
  const keepFalse = await scalarBool(false);

  const lrT = await cuda.h2d(new Float32Array([args.lr]), [], { dtype: 'float32', device: 0 });
  const epsT = await cuda.h2d(new Float32Array([1e-9]), [], { dtype: 'float32', device: 0 });

  const wInit = new Float32Array(1 * 10 * imgSize);
  for (let i = 0; i < wInit.length; i++) {
    wInit[i] = (Math.random() * 2 - 1) * 0.01;
  }
  let w = await cuda.h2d(wInit, [1, 10, imgSize], { dtype: 'float32', device: 0 });
  let b = await cuda.h2d(new Float32Array(1 * 10 * 1), [1, 10, 1], { dtype: 'float32', device: 0 });

  let globalStep = 0;

  for (let epoch = 0; epoch < args.epochs; epoch++) {
    for (let start = 0; start < mnist.train.count; start += B) {
      const batch = makeBatch(trainImages, trainLabels, start, B);
      if (batch.n === 0) break;

      const invB = await cuda.h2d(new Float32Array([1.0 / batch.n]), [], { dtype: 'float32', device: 0 });

      const x = await cuda.h2d(batch.x, [batch.n, 10, imgSize], { dtype: 'float32', device: 0 });
      const y = await cuda.h2d(batch.y, [batch.n, 10, 1], { dtype: 'float32', device: 0 });

      const xw = await ops.vt.mul(x, w);
      const logits0 = await ops.vt.sumDim(xw, dim2, keepTrue);
      const logits = await ops.vt.add(logits0, b);

      const maxLogits = await ops.vt.maxDim(logits, dim1, keepTrue);
      const centered = await ops.vt.sub(logits, maxLogits);
      const expLogits = await ops.vt.exp(centered);
      const sumExp = await ops.vt.sumDim(expLogits, dim1, keepTrue);
      const probs = await ops.vt.div(expLogits, sumExp);

      const probsSafe = await ops.vt.add(probs, epsT);
      const logProbs = await ops.vt.log(probsSafe);
      const nll = await ops.vt.mul(y, logProbs);
      const perExNll = await ops.vt.sumDim(nll, dim1, keepTrue);
      const negPerEx = await ops.vt.neg(perExNll);
      const lossSum = await ops.vt.sumDim(negPerEx, dim0, keepFalse);
      const loss = await ops.vt.mul(lossSum, invB);

      const probsMinusY = await ops.vt.sub(probs, y);
      const gradLogits = await ops.vt.mul(probsMinusY, invB);

      const gradW0 = await ops.vt.mul(gradLogits, x);
      const gradW = await ops.vt.sumDim(gradW0, dim0, keepFalse);

      const lrGradW = await ops.vt.mul(gradW, lrT);
      w = await ops.vt.sub(w, lrGradW);

      const gradB = await ops.vt.sumDim(gradLogits, dim0, keepFalse);
      const lrGradB = await ops.vt.mul(gradB, lrT);
      b = await ops.vt.sub(b, lrGradB);

      globalStep += 1;

      if (args.logEvery > 0 && globalStep % args.logEvery === 0) {
        const lossHost = await cuda.d2h(loss);
        const lossVal = /** @type {Float32Array} */ (lossHost)[0];

        const probsHost = await cuda.d2h(probs);
        const probsArr = /** @type {Float32Array} */ (probsHost);
        let correct = 0;
        for (let i = 0; i < batch.n; i++) {
          const off = i * 10;
          const pred = argMax10(probsArr.subarray(off, off + 10));
          if (pred === batch.yIdx[i]) correct += 1;
        }
        const acc = correct / batch.n;

        console.log(
          `epoch=${epoch} step=${globalStep} loss=${lossVal.toFixed(4)} acc=${acc.toFixed(3)}`,
        );
      }

      if (args.maxSteps > 0 && globalStep >= args.maxSteps) return;
    }
  }
}

main().catch((err) => {
  console.error(err?.stack ?? String(err));
  process.exitCode = 1;
});

