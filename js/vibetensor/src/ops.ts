// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as core from './core.js';
import { Tensor } from './tensor.js';

function toHandle(t: Tensor): core.TensorHandle {
  return t.__vbt_handle;
}

async function callUnary(name: string, a: Tensor): Promise<Tensor> {
  const handle = await core.callOp(name, [toHandle(a)]);
  return new Tensor(handle as core.TensorHandle);
}

async function callBinary(name: string, a: Tensor, b: Tensor): Promise<Tensor> {
  const handle = await core.callOp(name, [toHandle(a), toHandle(b)]);
  return new Tensor(handle as core.TensorHandle);
}

async function callTernary(
  name: string,
  a: Tensor,
  b: Tensor,
  c: Tensor,
): Promise<Tensor> {
  const handle = await core.callOp(name, [toHandle(a), toHandle(b), toHandle(c)]);
  return new Tensor(handle as core.TensorHandle);
}

export const vt = {
  async add(a: Tensor, b: Tensor): Promise<Tensor> {
    return callBinary('vt::add', a, b);
  },

  async mul(a: Tensor, b: Tensor): Promise<Tensor> {
    return callBinary('vt::mul', a, b);
  },

  async sub(a: Tensor, b: Tensor): Promise<Tensor> {
    return callBinary('vt::sub', a, b);
  },

  async div(a: Tensor, b: Tensor): Promise<Tensor> {
    return callBinary('vt::div', a, b);
  },

  async relu(a: Tensor): Promise<Tensor> {
    return callUnary('vt::relu', a);
  },

  async neg(a: Tensor): Promise<Tensor> {
    return callUnary('vt::neg', a);
  },

  async exp(a: Tensor): Promise<Tensor> {
    return callUnary('vt::exp', a);
  },

  async log(a: Tensor): Promise<Tensor> {
    return callUnary('vt::log', a);
  },

  async sumDim(self: Tensor, dim: Tensor, keepdim: Tensor): Promise<Tensor> {
    return callTernary('vt::sum_dim', self, dim, keepdim);
  },

  async maxDim(self: Tensor, dim: Tensor, keepdim: Tensor): Promise<Tensor> {
    return callTernary('vt::max_dim', self, dim, keepdim);
  },
};

export const ops = { vt };
