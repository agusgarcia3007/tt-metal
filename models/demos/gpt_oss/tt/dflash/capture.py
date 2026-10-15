# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The drafter's context features from the gpt-oss target: the outputs of target layers 1, 6, 11, 16, 21 (each the input
residual of the next layer, DecoderLayer.capture_input), read back to the host as they are produced (eager)."""

import torch

import ttnn

from ..fused_decode.boundary import tile_rows_to_flat


class TargetCapture:
    def __init__(self, model, layer_ids):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.hidden = model.hf_config.hidden_size
        self.prefill = {L: [] for L in self.layer_ids}
        self.last = {}
        for L in self.layer_ids:
            model.layers[L + 1].capture_input = lambda t, kind, L=L: self._record(L, t, kind)

    def remove(self):
        for L in self.layer_ids:
            self.model.layers[L + 1].capture_input = None

    def _host(self, t):
        devs = ttnn.get_device_tensors(t)
        x = ttnn.to_torch(devs[0]).float()
        if x.shape[-1] != self.hidden and x.shape[-1] * len(devs) == self.hidden:
            x = torch.cat([ttnn.to_torch(d).float() for d in devs], dim=-1)
        return x

    def _record(self, L, t, kind):
        if kind == "prefill":
            x = self._host(t)
            self.prefill[L].append(x.reshape(-1, x.shape[-1]))
        elif kind == "flat":
            self.last[L] = tile_rows_to_flat(ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0], self.hidden)[
                None
            ]
        else:  # rows: one row per user
            self.last[L] = self._host(t).reshape(-1, self.hidden)

    def prefill_features(self, n):
        """[n, len(layer_ids) * hidden] for the first n prompt positions; clears the prefill record."""
        out = torch.cat([torch.cat(self.prefill[L])[:n] for L in self.layer_ids], dim=-1)
        self.prefill = {L: [] for L in self.layer_ids}
        return out

    def decode_features(self):
        """[rows, len(layer_ids) * hidden] of the last decode step (1 row in the fused one-user path)."""
        return torch.cat([self.last[L] for L in self.layer_ids], dim=-1)
