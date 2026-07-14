/**
 * FlashAttention Tiling Data Structure
 * For AscendC (CANN 9.0.0) — vLLM compatibility on Ascend NPU.
 *
 * Implements tiled FlashAttention with online softmax.
 * Multi-core: split across heads and/or seq dimensions.
 *
 * Data layout (all half precision):
 *   Q: [batch, numHeads, seqLenQ, headDim]
 *   K: [batch, numKvHeads, seqLenK, headDim]
 *   V: [batch, numKvHeads, seqLenK, headDim]
 * Out: [batch, numHeads, seqLenQ, headDim]
 *
 * For GQA/MQA: numKvHeads < numHeads, each KV head is shared
 * by (numHeads / numKvHeads) query heads.
 */

#ifndef FLASH_ATTENTION_TILING_H
#define FLASH_ATTENTION_TILING_H

#include <cstdint>

#pragma pack(push, 8)
struct FlashAttentionTilingData {
    uint32_t batch;            // B: batch size
    uint32_t numHeads;         // H_q: number of query heads
    uint32_t numKvHeads;       // H_kv: number of key/value heads
    uint32_t seqLenQ;          // N_q: query sequence length
    uint32_t seqLenK;          // N_k: key/value sequence length
    uint32_t headDim;          // D: head dimension
    float    scale;            // 1.0f / sqrt(headDim)
    uint32_t qBlockSize;       // B_r: query tile size (tokens per step)
    uint32_t kvBlockSize;      // B_c: key/value block size
    uint32_t headsPerBlock;    // heads per multi-core block
    uint32_t isCausal;         // 1 = causal mask (lower triangular)
    uint32_t blockSplitMode;   // 0=heads-only, 1=heads+seq2D
    uint32_t workspaceSize;    // GM workspace size (bytes)
    uint32_t tilingQKOffset;   // byte offset to TCubeTiling for Q@K^T
    uint32_t tilingPVOffset;   // byte offset to TCubeTiling for P@V
};
#pragma pack(pop)

#endif // FLASH_ATTENTION_TILING_H
