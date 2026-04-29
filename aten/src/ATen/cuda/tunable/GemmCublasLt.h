#pragma once

#ifndef USE_ROCM

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <c10/util/StringUtil.h>
#include <fmt/printf.h>

#include <cstdlib>
#include <vector>

namespace at::cuda::tunable {

// Env-configurable number of heuristic algorithms to request from cuBLASLt.
// Increase this to try more candidates during autotuning.
// PYTORCH_TUNABLEOP_CUBLASLT_ALGO_COUNT=N
inline int GetCublasLtAlgoCount() {
  static int count = []() {
    const char* env = std::getenv("PYTORCH_TUNABLEOP_CUBLASLT_ALGO_COUNT");
    if (env != nullptr) {
      int v = std::atoi(env);
      if (v > 0) return v;
    }
    return 8;
  }();
  return count;
}

// Determines cuBLASLt data/compute types from template type T.
// Returns false if T is not supported.
template <typename T>
bool GetCublasLtTypes(
    cudaDataType_t& ab_type,
    cudaDataType_t& c_type,
    cublasComputeType_t& compute_type,
    cudaDataType_t& scale_type) {
  if constexpr (std::is_same_v<T, double>) {
    ab_type = CUDA_R_64F; c_type = CUDA_R_64F;
    compute_type = CUBLAS_COMPUTE_64F; scale_type = CUDA_R_64F;
  } else if constexpr (std::is_same_v<T, float>) {
    ab_type = CUDA_R_32F; c_type = CUDA_R_32F;
    compute_type = at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) == at::Float32Precision::TF32
                   ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    scale_type = CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, at::Half>) {
    ab_type = CUDA_R_16F; c_type = CUDA_R_16F;
    compute_type = CUBLAS_COMPUTE_32F; scale_type = CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, at::BFloat16>) {
    ab_type = CUDA_R_16BF; c_type = CUDA_R_16BF;
    compute_type = CUBLAS_COMPUTE_32F; scale_type = CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, c10::complex<float>>) {
    ab_type = CUDA_C_32F; c_type = CUDA_C_32F;
    compute_type = CUBLAS_COMPUTE_32F; scale_type = CUDA_C_32F;
  } else if constexpr (std::is_same_v<T, c10::complex<double>>) {
    ab_type = CUDA_C_64F; c_type = CUDA_C_64F;
    compute_type = CUBLAS_COMPUTE_64F; scale_type = CUDA_C_64F;
  } else {
    return false;
  }
  return true;
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
class CublasltGemmOp : public Callable<GemmParams<T>> {
 public:
  explicit CublasltGemmOp(cublasLtMatmulAlgo_t algo) : algo_(algo) {}

  TuningStatus Call(const GemmParams<T>* params) override {
    cudaDataType_t ab_type, c_type;
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type;
    if (!GetCublasLtTypes<T>(ab_type, c_type, compute_type, scale_type)) {
      return FAIL;
    }

    int64_t lda = params->lda, ldb = params->ldb, ldc = params->ldc;
    at::cuda::blas::_cublasAdjustLdLevel3(
        params->transa, params->transb, params->m, params->n, params->k,
        &lda, &ldb, &ldc);

    cublasOperation_t opa = at::cuda::blas::_cublasOpFromChar(params->transa);
    cublasOperation_t opb = at::cuda::blas::_cublasOpFromChar(params->transb);

    at::cuda::blas::CuBlasLtMatmulDescriptor op_desc(compute_type, scale_type);
    op_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
    op_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);

    at::cuda::blas::CuBlasLtMatrixLayout a_desc(ab_type, params->m, params->k, lda, opa != CUBLAS_OP_N);
    at::cuda::blas::CuBlasLtMatrixLayout b_desc(ab_type, params->k, params->n, ldb, opb != CUBLAS_OP_N);
    at::cuda::blas::CuBlasLtMatrixLayout c_desc(c_type, params->m, params->n, ldc);

    size_t workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
    void* workspace = at::cuda::getCUDABlasLtWorkspace();

    cublasLtMatmulHeuristicResult_t algo_check{};
    auto check_status = cublasLtMatmulAlgoCheck(
        at::cuda::getCurrentCUDABlasLtHandle(),
        op_desc.descriptor(),
        a_desc.descriptor(),
        b_desc.descriptor(),
        c_desc.descriptor(),
        c_desc.descriptor(),
        &algo_,
        &algo_check);
    if (check_status != CUBLAS_STATUS_SUCCESS) {
      return FAIL;
    }
    if (algo_check.workspaceSize > workspace_size) {
      return FAIL;
    }

    auto alpha = params->alpha;
    auto beta = params->beta;
    auto run_status = cublasLtMatmul(
        at::cuda::getCurrentCUDABlasLtHandle(),
        op_desc.descriptor(),
        &alpha,
        params->a, a_desc.descriptor(),
        params->b, b_desc.descriptor(),
        &beta,
        params->c, c_desc.descriptor(),
        params->c, c_desc.descriptor(),
        &algo_,
        workspace,
        workspace_size,
        at::cuda::getCurrentCUDAStream());

    return run_status == CUBLAS_STATUS_SUCCESS ? OK : FAIL;
  }

 private:
  cublasLtMatmulAlgo_t algo_;
};

template <typename T, BlasOp ALayout, BlasOp BLayout>
std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmParams<T>>>>>
GetCublasLtGemmTypeStringAndOps(const GemmParams<T>* params) {
  cudaDataType_t ab_type, c_type;
  cublasComputeType_t compute_type;
  cudaDataType_t scale_type;
  if (!GetCublasLtTypes<T>(ab_type, c_type, compute_type, scale_type)) {
    return {};
  }

  int64_t lda = params->lda, ldb = params->ldb, ldc = params->ldc;
  at::cuda::blas::_cublasAdjustLdLevel3(
      params->transa, params->transb, params->m, params->n, params->k,
      &lda, &ldb, &ldc);

  cublasOperation_t opa = at::cuda::blas::_cublasOpFromChar(params->transa);
  cublasOperation_t opb = at::cuda::blas::_cublasOpFromChar(params->transb);

  at::cuda::blas::CuBlasLtMatmulDescriptor op_desc(compute_type, scale_type);
  op_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
  op_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);

  at::cuda::blas::CuBlasLtMatrixLayout a_desc(ab_type, params->m, params->k, lda, opa != CUBLAS_OP_N);
  at::cuda::blas::CuBlasLtMatrixLayout b_desc(ab_type, params->k, params->n, ldb, opb != CUBLAS_OP_N);
  at::cuda::blas::CuBlasLtMatrixLayout c_desc(c_type, params->m, params->n, ldc);

  at::cuda::blas::CuBlasLtMatmulPreference preference;
  size_t workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size);

  int algo_count = GetCublasLtAlgoCount();
  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(algo_count);
  int returned_algos = 0;

  auto status = cublasLtMatmulAlgoGetHeuristic(
      at::cuda::getCurrentCUDABlasLtHandle(),
      op_desc.descriptor(),
      a_desc.descriptor(),
      b_desc.descriptor(),
      c_desc.descriptor(),
      c_desc.descriptor(),
      preference.descriptor(),
      algo_count,
      heuristic_results.data(),
      &returned_algos);

  if (status != CUBLAS_STATUS_SUCCESS || returned_algos == 0) {
    return {};
  }

  std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmParams<T>>>>> result;
  for (int i = 0; i < returned_algos; i++) {
    auto name = fmt::sprintf("Gemm_Cublaslt_%c_%c_%d",
        BlasOpToString(ALayout), BlasOpToString(BLayout), i);
    result.emplace_back(
        name,
        std::make_unique<CublasltGemmOp<T, ALayout, BLayout>>(heuristic_results[i].algo));
  }
  return result;
}

} // namespace at::cuda::tunable

#endif // !USE_ROCM
