## MNIST training (Node.js) using VibeTensor

This example trains a softmax regression model on MNIST using the **VibeTensor / vibetorch** runtime via the Node.js bindings.

### Environment setup

- Build the repo with the Node addon enabled (produces `js/vibetensor/vbt_napi.node`).
  - If your environment sets `CUDACXX` to a non-existent path, set it to your real `nvcc` (for example: `export CUDACXX=/usr/local/cuda/bin/nvcc`).
  - If CMake can’t find `node_api.h`, set `NODEJS_INCLUDE_DIR` to your Node headers (for example: `export NODEJS_INCLUDE_DIR="$(dirname "$(dirname "$(node -p 'process.execPath')")")/include/node"`).

- Build the TypeScript overlay:

```bash
cd js/vibetensor
npm install
npm run build
```

### Run

```bash
cd examples/mnist_nodejs
npm install
MNIST_BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist" MNIST_DIR="$(pwd)/.cache/mnist" npm run train:mnist
```

### Notes

- By default the script downloads MNIST into `examples/mnist_nodejs/.cache/mnist/`.
- If your environment has no outbound internet, set `MNIST_DIR` to a directory containing MNIST and set `MNIST_NO_DOWNLOAD=1`.
- CUDA is required for this example.

