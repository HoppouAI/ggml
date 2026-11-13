#include "im2col.cuh"

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t N, int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int tid = threadIdx.x;

    const int64_t total_channels = IC * KH * KW;
    const int threads_per_pos = blockDim.x;
    const int64_t start_ch = tid;
    const int64_t stride_ch = threads_per_pos;

    for (int64_t iow = blockIdx.x; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.y; iz < N_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t in = iz / OH;
            const int64_t ioh = iz - in * OH;

            for (int64_t iic_khw = start_ch; iic_khw < total_channels; iic_khw += stride_ch) {
                const int64_t iic = iic_khw / KH_KW;
                const int64_t rem = iic_khw - iic * KH_KW;
                const int64_t ikh = rem / KW;
                const int64_t ikw = rem - ikh * KW;

                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                const int64_t iih = ioh * s1 + ikh * d1 - p1;

                const int64_t offset_dst =
                    ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                    dst[offset_dst] = 0.0f;
                } else {
                    const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                    dst[offset_dst] = x[offset_src + iih * IW + iiw];
                }
            }
        }
    }
}

#define IM2COL_1D_TW 64
#define IM2COL_1D_CB 32

// 1d im2col through a shared memory tile, reads coalesce along time and writes along the output row
template <typename T, int KW_T>
static __global__ void im2col_1d_tiled_kernel(
        const float * x, T * dst, int64_t IC, int64_t IW, int64_t OW, int KW_rt,
        int64_t ch_stride, int64_t batch_stride, int s0, int p0, int d0, int tw_in) {
    extern __shared__ float tile[];
    const int KW = KW_T > 0 ? KW_T : KW_rt;
    const int64_t ow0 = (int64_t) blockIdx.x * IM2COL_1D_TW;
    const int64_t ic0 = (int64_t) blockIdx.y * IM2COL_1D_CB;
    const int64_t n   = blockIdx.z;
    const int64_t iw0 = ow0 * s0 - p0;
    const float * xn  = x + n * batch_stride;

    for (int i = threadIdx.x; i < IM2COL_1D_CB * tw_in; i += blockDim.x) {
        const int     c  = i / tw_in;
        const int     j  = i - c * tw_in;
        const int64_t ic = ic0 + c;
        const int64_t iw = iw0 + j;
        tile[i] = (ic < IC && iw >= 0 && iw < IW) ? xn[ic * ch_stride + iw] : 0.0f;
    }
    __syncthreads();

    const int     cbk    = IM2COL_1D_CB * KW;
    const int64_t rowlen = IC * KW;
    if constexpr (std::is_same_v<T, half>) {
        // half2 stores, a full channel block keeps every row segment 4 byte aligned
        if (ic0 + IM2COL_1D_CB <= IC && (rowlen & 1) == 0) {
            const int pairs = cbk / 2;
            for (int i = threadIdx.x; i < IM2COL_1D_TW * pairs; i += blockDim.x) {
                const int     o  = i / pairs;
                const int     e  = (i - o * pairs) * 2;
                const int64_t ow = ow0 + o;
                if (ow >= OW) {
                    continue;
                }
                const int c0 = e / KW, k0 = e - c0 * KW;
                const int c1 = (e + 1) / KW, k1 = e + 1 - c1 * KW;
                const float * t = tile + o * s0;
                half2 v = __floats2half2_rn(t[c0 * tw_in + k0 * d0], t[c1 * tw_in + k1 * d0]);
                *(half2 *) (dst + (n * OW + ow) * rowlen + ic0 * KW + e) = v;
            }
            return;
        }
    }
    for (int i = threadIdx.x; i < IM2COL_1D_TW * cbk; i += blockDim.x) {
        const int     o  = i / cbk;
        const int     r  = i - o * cbk;
        const int     c  = r / KW;
        const int     k  = r - c * KW;
        const int64_t ow = ow0 + o;
        const int64_t ic = ic0 + c;
        if (ow < OW && ic < IC) {
            dst[(n * OW + ow) * rowlen + ic * KW + k] = tile[c * tw_in + o * s0 + k * d0];
        }
    }
}

template <typename T>
static bool im2col_1d_tiled_cuda(const float * x, T * dst, int64_t IW, int64_t OW, int64_t KW, int64_t IC, int64_t N,
                                 int64_t ch_stride, int64_t batch_stride, int s0, int p0, int d0, cudaStream_t stream) {
    const int64_t tw_in = (int64_t) (IM2COL_1D_TW - 1) * s0 + (KW - 1) * d0 + 1;
    const size_t  smem  = (size_t) IM2COL_1D_CB * tw_in * sizeof(float);
    if (smem > 48 * 1024 || N > MAX_GRIDDIM_Z || KW > 4096) {
        return false;
    }
    dim3 grid((OW + IM2COL_1D_TW - 1) / IM2COL_1D_TW, (IC + IM2COL_1D_CB - 1) / IM2COL_1D_CB, N);
    switch (KW) {
        case 1:
            im2col_1d_tiled_kernel<T, 1><<<grid, 256, smem, stream>>>(x, dst, IC, IW, OW, 1, ch_stride, batch_stride, s0,
                                                                      p0, d0, (int) tw_in);
            break;
        case 7:
            im2col_1d_tiled_kernel<T, 7><<<grid, 256, smem, stream>>>(x, dst, IC, IW, OW, 7, ch_stride, batch_stride, s0,
                                                                      p0, d0, (int) tw_in);
            break;
        default:
            im2col_1d_tiled_kernel<T, 0><<<grid, 256, smem, stream>>>(x, dst, IC, IW, OW, (int) KW, ch_stride,
                                                                      batch_stride, s0, p0, d0, (int) tw_in);
            break;
    }
    return true;
}

static bool im2col_legacy() {
    static const bool legacy = [] {
        const char * v = getenv("GGML_CUDA_CONV1D_LEGACY");
        return v && v[0] && v[0] != '0';
    }();
    return legacy;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t N_OH = N * OH;
    const int64_t KH_KW = KW*KH;
    const int threads_per_block = MIN((int)IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE);
    dim3 block_nums(MIN(OW, MAX_GRIDDIM_Y), MIN(N_OH, MAX_GRIDDIM_Z));

    im2col_kernel<<<block_nums, threads_per_block, 0, stream>>>(
        x, dst, IC, IW, IH, OH, OW, KW, KH,
        N, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
        s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if (!is_2D && !im2col_legacy()) {
        const bool done = dst->type == GGML_TYPE_F16 ?
            im2col_1d_tiled_cuda(src1_d, (half *) dst_d, IW, OW, KW, IC, N, IC_IH_IW, IH_IW, s0, p0, d0, stream) :
            im2col_1d_tiled_cuda(src1_d, (float *) dst_d, IW, OW, KW, IC, N, IC_IH_IW, IH_IW, s0, p0, d0, stream);
        if (done) {
            return;
        }
    }

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static  __global__ void im2col_3d_kernel(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KD_KH_KW) {
        return;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);

    const int64_t iic = i / KD_KH_KW;
    const int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    const int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    const int64_t ikw = i % KW;

    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t in  = iz / OD_OH;
            const int64_t iod = (iz - in*OD_OH) / OH;
            const int64_t ioh = iz % OH;

            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t iid = iod * s2 + ikd * d2 - p2;

            const int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
                dst[offset_dst] = src[offset_src];
            }
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void im2col_3d_cuda(const float * src, T* dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t ID_IH_IW = ID*IH*IW;
    const int64_t KH_KW = KH*KW;
    const int64_t IH_IW = IH*IW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    const int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    const int64_t N_OD_OH = N*OD*OH;
    const int64_t OD_OH = OD*OH;
    const int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    const int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    const int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    const int64_t num_blocks = (IC_KD_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OD_OH, MAX_GRIDDIM_Z));
    im2col_3d_kernel<<<block_nums, MIN(IC_KD_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                                                                                           OH_OW, KD_KH_KW, ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW,
                                                                                           IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW,
                                                                                           OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                                                                           stride_q, stride_z, stride_y, stride_x,
                                                                                           s0, s1, s2, p0, p1, p2, d0, d1, d2);
}

static void im2col_3d_cuda_f16(const float * src, half * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                         stride_q, stride_z, stride_y, stride_x,
                         s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

static void im2col_3d_cuda_f32(const float * src, float * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<float>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const size_t  es       = ggml_element_size(src1);
    const int64_t stride_x = src1->nb[0] / es;
    const int64_t stride_y = src1->nb[1] / es;
    const int64_t stride_z = src1->nb[2] / es;
    const int64_t stride_q = src1->nb[3] / es;

    if(dst->type == GGML_TYPE_F16) {
        im2col_3d_cuda_f16(src1_d, (half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } else {
        im2col_3d_cuda_f32(src1_d, (float *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
