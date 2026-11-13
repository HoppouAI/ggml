#include "col2im-1d.cuh"
#include "convert.cuh"

// col2im_1d: scatter-add GEMM columns to 1D signal (gather approach)
// columns: [K*OC, T_in]  ->  output: [T_out, OC]
// Supports F32, F16, BF16 data with F32 accumulator.

template <typename T>
static __global__ void col2im_1d_kernel(
        const T * __restrict__ col,
        T       * __restrict__ dst,
        const int T_in, const uint3 T_out_fd,
        const int OC, const int K, const int K_OC,
        const int s0, const int p0, const int total) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total) return;

    // dst layout: [T_out, OC], ne[0]=T_out fastest
    const uint2 qr  = fast_div_modulo((uint32_t)idx, T_out_fd);  // qr.x = idx / T_out, qr.y = idx % T_out
    const int oc    = (int)qr.x;
    const int t_out = (int)qr.y;
    const int t_abs = t_out + p0;  // absolute position in uncropped signal

    // Gather: find all (t_in, k) where t_in*s + k == t_abs, 0 <= k < K
    int t_in_min = (t_abs - K + s0) / s0;  // ceil((t_abs - K + 1) / s)
    if (t_in_min < 0) t_in_min = 0;
    int t_in_max = t_abs / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
        const int k = t_abs - t_in * s0;
        // col layout: [K*OC, T_in], column index = oc * K + k
        sum += ggml_cuda_cast<float>(col[(oc * K + k) + t_in * K_OC]);
    }

    dst[idx] = ggml_cuda_cast<T>(sum);
}

#define COL2IM_1D_TW  64
#define COL2IM_1D_OCB 32

// f32 col2im with the needed col rows staged in shared memory, sums in the same order as the plain kernel
static __global__ void col2im_1d_tiled_f32(
        const float * __restrict__ col, float * __restrict__ dst,
        const int T_in, const int T_out, const int OC, const int K, const int K_OC, const int s0, const int p0) {
    extern __shared__ float tile[];
    const int t0    = blockIdx.x * COL2IM_1D_TW;
    const int oc0   = blockIdx.y * COL2IM_1D_OCB;
    const int ocn   = min(COL2IM_1D_OCB, OC - oc0);
    const int width = ocn * K;
    const int t_end = min(t0 + COL2IM_1D_TW, T_out) - 1;

    int tin_lo = (t0 + p0 - K + s0) / s0;
    if (tin_lo < 0) tin_lo = 0;
    int tin_hi = (t_end + p0) / s0;
    if (tin_hi >= T_in) tin_hi = T_in - 1;
    const int rows = tin_hi - tin_lo + 1;

    for (int i = threadIdx.x; i < rows * width; i += blockDim.x) {
        const int r = i / width;
        const int j = i - r * width;
        tile[i] = col[(int64_t) (tin_lo + r) * K_OC + oc0 * K + j];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < COL2IM_1D_TW * ocn; i += blockDim.x) {
        const int ocl   = i / COL2IM_1D_TW;
        const int t_out = t0 + (i - ocl * COL2IM_1D_TW);
        if (t_out >= T_out) {
            continue;
        }
        const int t_abs    = t_out + p0;
        int       t_in_min = (t_abs - K + s0) / s0;
        if (t_in_min < 0) t_in_min = 0;
        int t_in_max = t_abs / s0;
        if (t_in_max >= T_in) t_in_max = T_in - 1;

        float sum = 0.0f;
        for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
            const int k = t_abs - t_in * s0;
            sum += tile[(t_in - tin_lo) * width + ocl * K + k];
        }
        dst[(int64_t) (oc0 + ocl) * T_out + t_out] = sum;
    }
}

static bool conv1d_legacy() {
    static const bool legacy = [] {
        const char * v = getenv("GGML_CUDA_CONV1D_LEGACY");
        return v && v[0] && v[0] != '0';
    }();
    return legacy;
}

void ggml_cuda_op_col2im_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t OC = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];

    const int K_OC = (int) src0->ne[0];
    const int T_in = (int) src0->ne[1];
    const int K    = K_OC / OC;
    const int T_out = (int) dst->ne[0];

    if (src0->type == GGML_TYPE_F32 && !conv1d_legacy()) {
        const int    rows_max = (COL2IM_1D_TW - 1 + K) / s0 + 2;
        const size_t smem     = (size_t) rows_max * COL2IM_1D_OCB * K * sizeof(float);
        if (smem <= 48 * 1024) {
            dim3 grid((T_out + COL2IM_1D_TW - 1) / COL2IM_1D_TW, (OC + COL2IM_1D_OCB - 1) / COL2IM_1D_OCB);
            col2im_1d_tiled_f32<<<grid, 256, smem, stream>>>((const float *) src0->data, (float *) dst->data, T_in,
                                                              T_out, OC, K, K_OC, s0, p0);
            return;
        }
    }

    const uint3 T_out_fd = init_fastdiv_values((uint32_t)T_out);

    const int total = T_out * OC;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            col2im_1d_kernel<<<num_blocks, block_size, 0, stream>>>(
                (const float *)src0->data, (float *)dst->data,
                T_in, T_out_fd, OC, K, K_OC, s0, p0, total);
        } break;
        case GGML_TYPE_F16: {
            col2im_1d_kernel<<<num_blocks, block_size, 0, stream>>>(
                (const half *)src0->data, (half *)dst->data,
                T_in, T_out_fd, OC, K, K_OC, s0, p0, total);
        } break;
        case GGML_TYPE_BF16: {
            col2im_1d_kernel<<<num_blocks, block_size, 0, stream>>>(
                (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
                T_in, T_out_fd, OC, K, K_OC, s0, p0, total);
        } break;
        default:
            GGML_ABORT("col2im_1d: unsupported type");
    }
}
