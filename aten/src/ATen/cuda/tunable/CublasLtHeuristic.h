#pragma once

#include <ATen/cuda/CUDAContextLight.h>

#include <cstddef>

namespace at::cuda::tunable {

#ifndef USE_ROCM

cublasStatus_t getTunedCublasLtMatmulHeuristic(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t adesc,
    cublasLtMatrixLayout_t bdesc,
    cublasLtMatrixLayout_t cdesc,
    cublasLtMatrixLayout_t ddesc,
    cublasLtMatrixLayout_t heuristic_bdesc,
    cublasLtMatrixLayout_t heuristic_cdesc,
    cublasLtMatrixLayout_t heuristic_ddesc,
    cublasLtMatmulPreference_t preference,
    const void* alpha,
    const void* a,
    const void* b,
    const void* beta,
    const void* c,
    void* d,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream,
    cublasLtMatmulHeuristicResult_t* heuristic_result,
    int* returned_result);

#endif // USE_ROCM

} // namespace at::cuda::tunable
