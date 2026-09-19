// Probe for docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md
// TG-1: measures what the 17-byte ROCmI4 block layout costs vs 16-byte aligned.
// Build: hipcc -O3 --offload-arch=gfx1100 -o layoutprobe layoutprobe.hip
// Block = { uint8 qs[16]; uint8 e; } with no padding, so the 16-byte qs field
// starts at 17k and is never 16-byte aligned except at k=0. MMVQ walks a row
// with stride 17. This compares that walk against a 16-byte-aligned layout with
// the same useful bytes.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define BLOCKS_PER_ROW 512        // 512 * 32 = 16384 weights per row
#define NROWS 1

// A: 17-byte blocks, read 16 bytes per block at offset 17k+1 (MMVQ pattern)
__global__ void __launch_bounds__(256) k_packed(const uint8_t * __restrict__ buf, size_t nblk, unsigned * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i < nblk; i += stride) {
        uint4 v;
        __builtin_memcpy(&v, buf + i*17 + 1, 16);   // misaligned by 1 byte
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u)
        out[threadIdx.x & 7] = acc.x;
}

// B: same useful bytes, 16-byte aligned blocks
__global__ void __launch_bounds__(256) k_aligned(const uint8_t * __restrict__ buf, size_t nblk, unsigned * out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i < nblk; i += stride) {
        uint4 v;
        __builtin_memcpy(&v, buf + i*16, 16);       // aligned
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xFEEDFACEu && acc.z == 0x12345678u && acc.w == 0x9ABCDEF0u)
        out[threadIdx.x & 7] = acc.x;
}

int main(void) {
    const size_t nblk = getenv("NBLK") ? strtoull(getenv("NBLK"), 0, 10) : (size_t)250000000;
    const size_t sz_p = nblk * 17, sz_a = nblk * 16;
    uint8_t *bp, *ba; unsigned *out;
    hipMalloc(&bp, sz_p); hipMalloc(&ba, sz_a); hipMalloc(&out, 4096);
    hipMemset(bp, 1, sz_p); hipMemset(ba, 1, sz_a);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);

    auto bench = [&](const char * name, void (*k)(const uint8_t*, size_t, unsigned*), uint8_t * buf, size_t useful) {
        int blocks = 1536, threads = 256;
        k<<<blocks, threads>>>(buf, nblk, out); hipDeviceSynchronize();
        float best = 1e30f;
        for (int r = 0; r < 3; r++) {
            hipEventRecord(e0); k<<<blocks, threads>>>(buf, nblk, out); hipEventRecord(e1); hipEventSynchronize(e1);
            float t; hipEventElapsedTime(&t, e0, e1); if (t < best) best = t;
        }
        printf("%-9s nblk=%zu useful=%.2f GB : %7.1f GB/s  (%.3f ms)\n",
               name, nblk, useful/1e9, useful/(best*1e-3)/1e9, best);
    };
    bench("packed17", k_packed,  bp, (double)nblk*16);
    bench("aligned16", k_aligned, ba, (double)nblk*16);
    return 0;
}
