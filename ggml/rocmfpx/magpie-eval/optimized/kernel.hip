// Magpie evaluation harness for the ROCmI4 W4A4 MMQ vec-dot epilogue.
// The evaluated code path is the real ggml_cuda_mmq_vec_dot_rocmi4_w4a4_wmma
// plus the real tile loader from the ROCmFPX tree given by -I at compile
// time. Staging and iteration replicate mul_mat_q_process_tile for the
// J=128, I=64, 4-warp config on one 256-element K tile.
//
// Modes: default runs a correctness pass (grid 1, fp64 reference, tolerance,
// and a bit-exact checksum of all outputs) then a perf pass (large grid,
// event-timed us per process-tile iteration). Exit code is nonzero if the
// correctness pass fails.
//
// Correctness data uses one constant nibble value per (row, block) on both
// operands, so the fp64 reference does not depend on the nibble-to-k slot
// order inside a dword; that ordering is covered by the repo's IU4 oracle
// and test-backend-ops. Scales stay per (row, block) so scale indexing is
// still exercised.
#include "ggml-cuda/common.cuh"
#include "mmq.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

constexpr int J_T   = 128;
constexpr int I_T   = 64;
constexpr int NW    = 4;
constexpr int TPB   = NW * 32;
constexpr int SR    = 44;
constexpr int YK    = MMQ_TILE_Y_K;
constexpr int KB    = 8;                 // 32-element blocks per tile
constexpr int SUMS  = J_T * I_T / TPB;   // 64 per thread
constexpr int X_INTS = I_T * SR;
constexpr int Y_INTS = J_T * YK;
constexpr int SH_INTS = J_T + GGML_PAD(Y_INTS, TPB) + X_INTS;

__global__ void eval_mmq(const char * __restrict__ x, const int * __restrict__ y,
                         float * __restrict__ out, int iters) {
    constexpr ggml_type type = GGML_TYPE_Q4_0_ROCMI4;
    auto funcs = ggml_cuda_mmq_get_util_funcs<type, J_T, false>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J_T;
    int * tile_x = tile_y + GGML_PAD(Y_INTS, TPB);

    float sum[SUMS];
#pragma unroll
    for (int s = 0; s < SUMS; ++s) sum[s] = 0.0f;

    const int x_off = blockIdx.x * I_T * KB;

    for (int it = 0; it < iters; ++it) {
        funcs.load_tiles(x, tile_x, x_off, I_T - 1, KB);
        {
            const int * by0 = y + blockIdx.x * 2 * Y_INTS;
            for (int l0 = 0; l0 < Y_INTS; l0 += TPB) {
                tile_y[l0 + threadIdx.y*32 + threadIdx.x] = by0[l0 + threadIdx.y*32 + threadIdx.x];
            }
        }
        __syncthreads();
        funcs.vec_dot(tile_x, tile_y, sum, 0);
        __syncthreads();
        {
            const int * by0 = y + blockIdx.x * 2 * Y_INTS + Y_INTS;
            for (int l0 = 0; l0 < Y_INTS; l0 += TPB) {
                tile_y[l0 + threadIdx.y*32 + threadIdx.x] = by0[l0 + threadIdx.y*32 + threadIdx.x];
            }
        }
        __syncthreads();
        funcs.vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    float * o = out + (size_t) blockIdx.x * TPB * SUMS + (threadIdx.y*32 + threadIdx.x) * SUMS;
#pragma unroll
    for (int s = 0; s < SUMS; ++s) o[s] = sum[s];
}

// Decode UE4M3 scale bytes with the real device function so the host
// reference sees bit-identical scales.
__global__ void decode_da(const unsigned char * e, float * da, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) da[i] = rocmfpx_ue4m3_to_fp32_finite(e[i]);
}

// Exact fp64 reference for one output element. Constant nibbles per
// (row, block) make the integer dot per block exactly 32 * xv * yv.
static double ref(const int * xv, const int * yv, const float * da, const float * db, int i, int j) {
    double acc = 0.0;
    for (int blk = 0; blk < KB; ++blk) {
        const double dot = 32.0 * (double) xv[i*KB + blk] * (double) yv[j*KB + blk];
        acc += dot * 16.0 * (double) da[i*KB + blk] * (double) db[j*KB + blk];
    }
    return acc;
}

int main(int argc, char ** argv) {
    const bool perf = (argc > 1 && strcmp(argv[1], "--perf") == 0);

    std::mt19937 rng(20260919);
    std::uniform_int_distribution<int> dword_dist(0, 0x7FFFFFFF);

    if (!perf) {
        std::vector<int> x_qs(I_T * SR);
        std::vector<unsigned char> x_e(I_T * KB);
        std::vector<int> y_seg(2 * Y_INTS);
        std::uniform_int_distribution<int> nib_dist(-8, 7);
        std::vector<int> xv(I_T * KB), yv(J_T * KB);
        for (int i = 0; i < I_T; ++i) {
            for (int b = 0; b < KB; ++b) {
                const int v = nib_dist(rng);
                xv[i*KB + b] = v;
                // replicate the 4-bit two's-complement code into all nibbles
                const unsigned int word = (unsigned) (v & 0xF) * 0x11111111u;
                for (int q = 0; q < 4; ++q) x_qs[i*SR + 4*b + q] = (int) word;
                x_e[i*KB + b] = (unsigned char) (rng() & 0xFF);
            }
        }
        std::uniform_real_distribution<float> fdist(0.05f, 1.5f);
        std::vector<float> da(I_T * KB), db(J_T * KB);
        for (int i = 0; i < I_T; ++i)
            for (int b = 0; b < KB; ++b) da[i*KB + b] = fdist(rng);
        for (int j = 0; j < J_T; ++j)
            for (int b = 0; b < KB; ++b) db[j*KB + b] = fdist(rng);
        // y segment g: row ints [0..3] hold db bits for blocks 4g..4g+3;
        // ints [4..19] hold 16 packed dwords (8 elements each) covering the
        // segment's 128 elements; ints [20..35] are unused by the k loop.
        for (int g = 0; g < 2; ++g) {
            for (int j = 0; j < J_T; ++j) {
                int * row = y_seg.data() + g*Y_INTS + j*YK;
                for (int q = 0; q < 4; ++q) {
                    unsigned int bits; memcpy(&bits, &db[j*KB + 4*g + q], 4);
                    row[q] = (int) bits;
                }
                for (int q = 0; q < 16; ++q) {
                    const int v = nib_dist(rng);
                    yv[j*KB + 4*g + q/4] = v;
                    row[4 + q] = (int) ((unsigned) (v & 0xF) * 0x11111111u);
                }
                for (int q = 16; q < 32; ++q) row[4 + q] = 0;
            }
        }

        // Device x buffer must be the on-disk block_rocmi4 layout: 17-byte
        // records (four qs dwords + one UE4M3 scale byte), KB records per row.
        std::vector<char> x_disk((size_t) I_T * KB * 17);
        for (int i = 0; i < I_T; ++i) {
            for (int b = 0; b < KB; ++b) {
                char * rec = x_disk.data() + ((size_t) i*KB + b) * 17;
                memcpy(rec, &x_qs[i*SR + 4*b], 16);
                rec[16] = (char) x_e[i*KB + b];
            }
        }

        int * dx; int * dy; float * dout; unsigned char * de; float * dda;
        (void) hipMalloc(&dx, x_disk.size());
        (void) hipMalloc(&dy, y_seg.size() * 4);
        (void) hipMalloc(&dout, (size_t) TPB * SUMS * 4);
        (void) hipMalloc(&de, x_e.size());
        (void) hipMalloc(&dda, da.size() * 4);
        (void) hipMemcpy(dx, x_disk.data(), x_disk.size(), hipMemcpyHostToDevice);
        (void) hipMemcpy(dy, y_seg.data(), y_seg.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(de, x_e.data(), x_e.size(), hipMemcpyHostToDevice);
        decode_da<<<(I_T*KB + 255)/256, 256>>>(de, dda, I_T*KB);
        (void) hipMemcpy(da.data(), dda, da.size() * 4, hipMemcpyDeviceToHost);
        hipLaunchKernelGGL(eval_mmq, dim3(1), dim3(32, NW), SH_INTS*4, 0, (const char *) dx, dy, dout, 1);
        std::vector<float> out((size_t) TPB * SUMS);
        (void) hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost);

        // thread (ty, tx), slot l -> i = ty*16 + 2*l + tx/16, j = jn*16 + tx%16
        unsigned long long ck = 0;
        double worst = 0.0;
        bool ok = true;
        for (int ty = 0; ty < NW; ++ty)
        for (int l = 0; l < SUMS; ++l) {
            const int jn = l / 8;
            const int lc = l % 8;
            for (int tx = 0; tx < 32; ++tx) {
                const int i = ty*16 + 2*lc + tx/16;
                const int j = jn*16 + tx % 16;
                const float got = out[(size_t)(ty*32 + tx) * SUMS + l];
                const double want = ref(xv.data(), yv.data(), da.data(), db.data(), i, j);
                const double denom = fabs(want) + 1.0;
                const double err = fabs((double) got - want) / denom;
                worst = err > worst ? err : worst;
                if (err > 2e-3) ok = false;
                unsigned int bits; memcpy(&bits, &got, 4);
                ck = (ck * 1099511628211ULL) ^ bits;
            }
        }
        // Golden checksum of the merged-baseline outputs; the evaluated
        // variant must reproduce it bit-exactly.
        const unsigned long long want_ck = 0x64e99ade789a33b7ULL;
        printf("correctness: worst_rel_err=%.3e checksum=%016llx\n", worst, ck);
        if (ck != want_ck) {
            printf("checksum mismatch against gold %016llx\n", want_ck);
            return 1;
        }
        printf("bit-exact vs gold: PASS\n");
        (void) hipFree(dx); (void) hipFree(dy); (void) hipFree(dout); (void) hipFree(de); (void) hipFree(dda);
    }

    // ---------------- perf: large grid, event timed ----------------
    const int blocks = 1024;
    const int iters = 64;
    std::vector<char> x_buf((size_t) blocks * I_T * KB * 17);
    for (size_t k = 0; k < x_buf.size(); ++k) x_buf[k] = (char) (rng() & 0xFF);
    std::vector<int> y_buf((size_t) blocks * 2 * Y_INTS);
    for (size_t k = 0; k < y_buf.size(); ++k) y_buf[k] = (int) dword_dist(rng);
    float * dout;
    (void) hipMalloc(&dout, (size_t) blocks * TPB * SUMS * 4);

    char * dx; int * dy;
    (void) hipMalloc(&dx, x_buf.size());
    (void) hipMalloc(&dy, y_buf.size() * 4);
    (void) hipMemcpy(dx, x_buf.data(), x_buf.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(dy, y_buf.data(), y_buf.size() * 4, hipMemcpyHostToDevice);

    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipLaunchKernelGGL(eval_mmq, dim3(blocks), dim3(32, NW), SH_INTS*4, 0, (const char *) dx, dy, dout, iters);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
        hipEventRecord(e0);
        hipLaunchKernelGGL(eval_mmq, dim3(blocks), dim3(32, NW), SH_INTS*4, 0, (const char *) dx, dy, dout, iters);
        hipEventRecord(e1);
        hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    printf("perf: blocks=%d iters=%d best=%.3f ms -> %.2f us/tile-iteration\n",
           blocks, iters, best, best * 1000.0f / iters);
    (void) hipFree(dx); hipFree(dy); hipFree(dout);
    return 0;
}
