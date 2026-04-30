#include <ATen/native/cuda/CublasGroupedArgs.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

template <typename IndexType>
__global__ void populate_cublas_grouped_args_kernel(
    const int32_t* __restrict__ offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    IndexType cublas_m, IndexType cublas_n, IndexType cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    IndexType lda_val, IndexType ldb_val, IndexType ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    IndexType* __restrict__ m_out, IndexType* __restrict__ n_out, IndexType* __restrict__ k_out,
    IndexType* __restrict__ lda_out, IndexType* __restrict__ ldb_out, IndexType* __restrict__ ldd_out,
    int64_t* __restrict__ APtr_out, int64_t* __restrict__ BPtr_out, int64_t* __restrict__ DPtr_out,
    int64_t* __restrict__ alphaPtr_out, int64_t* __restrict__ betaPtr_out,
    float* __restrict__ alpha_ptr, float* __restrict__ beta_ptr) {
  int i = threadIdx.x;

  if (i == 0) {
    *alpha_ptr = 1.0f;
    *beta_ptr = 0.0f;
  }

  int32_t delta = 0;
  int64_t group_start = 0;
  if (offs != nullptr) {
    int32_t end = offs[i];
    int32_t start_val = (i == 0) ? 0 : offs[i - 1];
    delta = end - start_val;
    group_start = static_cast<int64_t>(start_val);
  }

  m_out[i] = m_is_delta ? static_cast<IndexType>(delta) : cublas_m;
  n_out[i] = n_is_delta ? static_cast<IndexType>(delta) : cublas_n;
  k_out[i] = k_is_delta ? static_cast<IndexType>(delta) : cublas_k;

  lda_out[i] = lda_val;
  ldb_out[i] = ldb_val;
  ldd_out[i] = ldd_val;

  APtr_out[i] = base_A + group_start * a_offs_stride + i * a_idx_stride;
  BPtr_out[i] = base_B + group_start * b_offs_stride + i * b_idx_stride;
  DPtr_out[i] = base_D + group_start * d_offs_stride + i * d_idx_stride;

  // Store device pointer as int64_t — cublasLt grouped GEMM expects
  // per-group alpha/beta as arrays of device pointers.
  alphaPtr_out[i] = reinterpret_cast<int64_t>(alpha_ptr);
  betaPtr_out[i] = reinterpret_cast<int64_t>(beta_ptr);
}

template <typename IndexType>
void launch_populate_cublas_grouped_args(
    int batchCount,
    const int32_t* offs,
    int64_t base_A, int64_t base_B, int64_t base_D,
    IndexType cublas_m, IndexType cublas_n, IndexType cublas_k,
    bool m_is_delta, bool n_is_delta, bool k_is_delta,
    IndexType lda_val, IndexType ldb_val, IndexType ldd_val,
    int64_t a_offs_stride, int64_t a_idx_stride,
    int64_t b_offs_stride, int64_t b_idx_stride,
    int64_t d_offs_stride, int64_t d_idx_stride,
    IndexType* m_out, IndexType* n_out, IndexType* k_out,
    IndexType* lda_out, IndexType* ldb_out, IndexType* ldd_out,
    int64_t* APtr_out, int64_t* BPtr_out, int64_t* DPtr_out,
    int64_t* alphaPtr_out, int64_t* betaPtr_out,
    float* alpha_ptr, float* beta_ptr,
    cudaStream_t stream) {
  populate_cublas_grouped_args_kernel<<<1, batchCount, 0, stream>>>(
      offs, base_A, base_B, base_D,
      cublas_m, cublas_n, cublas_k,
      m_is_delta, n_is_delta, k_is_delta,
      lda_val, ldb_val, ldd_val,
      a_offs_stride, a_idx_stride,
      b_offs_stride, b_idx_stride,
      d_offs_stride, d_idx_stride,
      m_out, n_out, k_out,
      lda_out, ldb_out, ldd_out,
      APtr_out, BPtr_out, DPtr_out,
      alphaPtr_out, betaPtr_out,
      alpha_ptr, beta_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
cublasGroupedArgs::cublasGroupedArgs(
    const Tensor& mat1,
    const Tensor& mat2,
    const std::optional<Tensor>& offs,
    Tensor& c,
    int batchCount_,
    bool needs_int64) {
  const bool a_is_2d = mat1.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  if (a_is_2d || b_is_2d) {
    TORCH_CHECK(offs.has_value(), "Offsets tensor must be provided when at least one input is 2D");
  }

  A_dtype = mat2.scalar_type();
  B_dtype = mat1.scalar_type();
  result_dtype = c.scalar_type();
  const int64_t esz = mat1.element_size();
  const int64_t out_esz = c.element_size();

  batchCount = batchCount_;
  use_int64 = needs_int64;

  // cuBLAS is column-major. To get a row-major result C = mat1 × mat2,
  // we use the identity C^T = mat2^T × mat1^T. So cuBLAS-A = mat2 and
  // cuBLAS-B = mat1. The transpose flags depend on inner-dim layout:
  //   row-major (stride(-1)==1): cuBLAS sees it as col-major "already
  //     transposed" → after the B^T×A^T flip, the op flag is 'n'
  //   col-major (stride(-2)==1): cuBLAS sees it naturally → after
  //     the flip, the op flag is 't'
  const bool mat2_row_major = mat2.stride(-1) == 1;
  const bool mat1_row_major = mat1.stride(-1) == 1;
  transa = mat2_row_major ? 'n' : 't';
  transb = mat1_row_major ? 'n' : 't';

  // User-space dimensions
  const int64_t user_M = mat1.size(-2);
  const int64_t user_N = mat2.size(-1);
  const int64_t user_K = mat1.size(-1);

  // In the cuBLAS B^T×A^T convention:
  //   cublas_m = user_N, cublas_n = user_M, cublas_k = user_K
  const int64_t cublas_m = user_N;
  const int64_t cublas_n = user_M;
  const int64_t cublas_k = user_K;

  // Leading dimensions (constant across groups, from inner-dim strides)
  // cuBLAS-A = mat2, cuBLAS-B = mat1
  const int64_t lda_val = transa == 't' ? mat2.stride(-1) : mat2.stride(-2);
  const int64_t ldb_val = transb == 't' ? mat1.stride(-1) : mat1.stride(-2);
  const int64_t ldd_val = c.stride(-2);

  // Determine per-case which dimensions are variable (delta-based)
  // and how pointer strides work
  bool m_is_delta = false, n_is_delta = false, k_is_delta = false;
  int64_t a_offs_stride = 0, a_idx_stride = 0;
  int64_t b_offs_stride = 0, b_idx_stride = 0;
  int64_t d_offs_stride = 0, d_idx_stride = 0;

  if (a_is_2d && b_is_2d) {
    // 2D x 2D: jagged K
    k_is_delta = true;
    a_offs_stride = mat2.stride(-2) * esz;
    b_offs_stride = mat1.stride(-1) * esz;
    d_idx_stride = c.stride(0) * out_esz;
    avgM = cublas_m;
    avgN = cublas_n;
    avgK = user_K / batchCount;
  } else if (a_is_2d && !b_is_2d) {
    // 2D x 3D: jagged M (user M varies, cublas n varies)
    n_is_delta = true;
    a_idx_stride = mat2.stride(0) * esz;
    b_offs_stride = mat1.stride(-2) * esz;
    d_offs_stride = c.stride(-2) * out_esz;
    avgM = cublas_m;
    avgN = user_M / batchCount;
    avgK = cublas_k;
  } else if (!a_is_2d && b_is_2d) {
    // 3D x 2D: jagged N (user N varies, cublas m varies)
    m_is_delta = true;
    a_offs_stride = mat2.stride(-1) * esz;
    b_idx_stride = mat1.stride(0) * esz;
    d_offs_stride = c.stride(-1) * out_esz;
    avgM = user_N / batchCount;
    avgN = cublas_n;
    avgK = cublas_k;
  } else {
    // 3D x 3D: all dimensions fixed
    a_idx_stride = mat2.stride(0) * esz;
    b_idx_stride = mat1.stride(0) * esz;
    d_idx_stride = c.stride(0) * out_esz;
    avgM = cublas_m;
    avgN = cublas_n;
    avgK = cublas_k;
  }

  // Determine element size for dimension arrays
  const size_t dim_elem_size = use_int64 ? sizeof(int64_t) : sizeof(int32_t);

  // Single device allocation for all arrays:
  //   6 x dim_elem_size[batchCount]  (m, n, k, lda, ldb, ldd)
  //   5 x int64[batchCount]          (A, B, D, alpha, beta ptrs)
  //   2 x float                      (alpha, beta scalars)
  const int64_t buf_bytes =
      static_cast<int64_t>(batchCount) * 6 * dim_elem_size +
      static_cast<int64_t>(batchCount) * 5 * sizeof(int64_t) +
      2 * sizeof(float);
  buf = at::empty({buf_bytes}, mat1.options().dtype(at::kByte));

  const int64_t base_A = reinterpret_cast<int64_t>(mat2.data_ptr());
  const int64_t base_B = reinterpret_cast<int64_t>(mat1.data_ptr());
  const int64_t base_D = reinterpret_cast<int64_t>(c.data_ptr());

  const int32_t* offs_ptr = offs.has_value()
      ? static_cast<const int32_t*>(offs.value().data_ptr())
      : nullptr;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_int64) {
    // Pack as int64_t arrays
    int64_t* m_arr = reinterpret_cast<int64_t*>(buf.data_ptr());
    int64_t* n_arr = m_arr + batchCount;
    int64_t* k_arr = n_arr + batchCount;
    int64_t* lda_arr = k_arr + batchCount;
    int64_t* ldb_arr = lda_arr + batchCount;
    int64_t* ldd_arr = ldb_arr + batchCount;

    APtrArray     = reinterpret_cast<int64_t*>(ldd_arr + batchCount);
    BPtrArray     = APtrArray + batchCount;
    DPtrArray     = BPtrArray + batchCount;
    alphaPtrArray = DPtrArray + batchCount;
    betaPtrArray  = alphaPtrArray + batchCount;

    float* alpha_scalar = reinterpret_cast<float*>(betaPtrArray + batchCount);
    float* beta_scalar  = alpha_scalar + 1;
    alphaScalar = alpha_scalar;
    betaScalar = beta_scalar;

    mArray = m_arr;
    nArray = n_arr;
    kArray = k_arr;
    ldaArray = lda_arr;
    ldbArray = ldb_arr;
    lddArray = ldd_arr;

    launch_populate_cublas_grouped_args<int64_t>(
        batchCount, offs_ptr,
        base_A, base_B, base_D,
        static_cast<int64_t>(cublas_m), static_cast<int64_t>(cublas_n), static_cast<int64_t>(cublas_k),
        m_is_delta, n_is_delta, k_is_delta,
        static_cast<int64_t>(lda_val), static_cast<int64_t>(ldb_val), static_cast<int64_t>(ldd_val),
        a_offs_stride, a_idx_stride,
        b_offs_stride, b_idx_stride,
        d_offs_stride, d_idx_stride,
        m_arr, n_arr, k_arr,
        lda_arr, ldb_arr, ldd_arr,
        APtrArray, BPtrArray, DPtrArray,
        alphaPtrArray, betaPtrArray,
        alpha_scalar, beta_scalar,
        stream);
  } else {
    // Pack as int32_t arrays (common case)
    int32_t* m_arr = reinterpret_cast<int32_t*>(buf.data_ptr());
    int32_t* n_arr = m_arr + batchCount;
    int32_t* k_arr = n_arr + batchCount;
    int32_t* lda_arr = k_arr + batchCount;
    int32_t* ldb_arr = lda_arr + batchCount;
    int32_t* ldd_arr = ldb_arr + batchCount;

    APtrArray     = reinterpret_cast<int64_t*>(ldd_arr + batchCount);
    BPtrArray     = APtrArray + batchCount;
    DPtrArray     = BPtrArray + batchCount;
    alphaPtrArray = DPtrArray + batchCount;
    betaPtrArray  = alphaPtrArray + batchCount;

    float* alpha_scalar = reinterpret_cast<float*>(betaPtrArray + batchCount);
    float* beta_scalar  = alpha_scalar + 1;
    alphaScalar = alpha_scalar;
    betaScalar = beta_scalar;

    mArray = m_arr;
    nArray = n_arr;
    kArray = k_arr;
    ldaArray = lda_arr;
    ldbArray = ldb_arr;
    lddArray = ldd_arr;

    launch_populate_cublas_grouped_args<int32_t>(
        batchCount, offs_ptr,
        base_A, base_B, base_D,
        static_cast<int32_t>(cublas_m), static_cast<int32_t>(cublas_n), static_cast<int32_t>(cublas_k),
        m_is_delta, n_is_delta, k_is_delta,
        static_cast<int32_t>(lda_val), static_cast<int32_t>(ldb_val), static_cast<int32_t>(ldd_val),
        a_offs_stride, a_idx_stride,
        b_offs_stride, b_idx_stride,
        d_offs_stride, d_idx_stride,
        m_arr, n_arr, k_arr,
        lda_arr, ldb_arr, ldd_arr,
        APtrArray, BPtrArray, DPtrArray,
        alphaPtrArray, betaPtrArray,
        alpha_scalar, beta_scalar,
        stream);
  }
}
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

} // namespace at::native
