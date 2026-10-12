// Small kernels of the DSpark drafter (model/dspark in forward_ref; reference: model.py DSparkBlock).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/ref.hpp"

namespace unholy::kernels {

// out[r, 0:D] = bf16(mean over the hc copies of x[rows[r]]), x [T, hc, D]; out row stride ldo.
void hc_mean_rows(const bf16* x, int hc, int D, const int32_t* rows, int R, bf16* out, int ldo, hipStream_t st);
// ring[slots[r]][pos[r] % 128][:] = kv[r][:], ring [slots][128][D].
void scatter_window(const bf16* kv, int R, int D, const int32_t* slots, const int32_t* pos, bf16* ring,
                    hipStream_t st);
// Markov step i of the draft head for R requests (logits rows r * B + i, B = block size, ld = vocab):
// e[r] = w1[tok[r * (B + 1) + i]] (saved to emb[r * B + i]), logits[r * B + i][v] += w2[v] . e[r] in
// fp32, then tok[r * (B + 1) + i + 1] = argmax_v (lowest index on ties).
// `part`: markov_scratch_bytes(V, R) of device scratch.
size_t markov_scratch_bytes(int V, int R);
void markov_step(const bf16* w1, const bf16* w2, int V, int K, float* logits, int B, int32_t* tok, bf16* emb,
                 int R, int i, void* part, hipStream_t st);
// conf[n] = w . [x[n], emb[n]] in fp32 (w bf16 [D + K]).
void dspark_confidence(const bf16* x, int D, const bf16* emb, int K, const bf16* w, float* conf, int N,
                       hipStream_t st);

// Speculative-decoding ring backup (StepInput::backup / RingRestore). A RingDesc is one per-slot ring
// buffer: slot s's ring at base + s * slot_bytes, position p's row at (p % ring) * row_bytes; `off` is the
// row's place in a backed-up position (pos_bytes = the sum of row_bytes). The backup of a slot holds the
// rows of positions [s0, s0 + n) as they were before the verify chunk overwrote them.
struct RingDesc {
  uint8_t* base;
  size_t slot_bytes;
  int ring, row_bytes;
  size_t off;
};
void ring_backup(const RingDesc* descs, int nd, int slot, int s0, int n, uint8_t* backup, size_t pos_bytes,
                 hipStream_t st);
// Puts back the rows of positions [from, to) from a backup taken at s0.
void ring_restore(const RingDesc* descs, int nd, int slot, int s0, int from, int to, const uint8_t* backup,
                  size_t pos_bytes, hipStream_t st);

}  // namespace unholy::kernels
