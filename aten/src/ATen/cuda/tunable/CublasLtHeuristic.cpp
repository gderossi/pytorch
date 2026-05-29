#include <ATen/cuda/tunable/CublasLtHeuristic.h>

#ifndef USE_ROCM

#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace at::cuda::tunable {
namespace {

constexpr int kCublasLtAutotuneRepeats = 5;

struct CudaEvent {
  CudaEvent() {
    AT_CUDA_CHECK(cudaEventCreate(&event));
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent() {
    if (event != nullptr) {
      C10_CUDA_CHECK_WARN(cudaEventDestroy(event));
    }
  }

  operator cudaEvent_t() const {
    return event;
  }

  cudaEvent_t event = nullptr;
};

template <typename T>
bool getMatmulDescAttribute(
    cublasLtMatmulDesc_t desc,
    cublasLtMatmulDescAttributes_t attr,
    T* value) {
  size_t written = 0;
  return cublasLtMatmulDescGetAttribute(
             desc, attr, value, sizeof(T), &written) == CUBLAS_STATUS_SUCCESS;
}

template <typename T>
bool getMatrixLayoutAttribute(
    cublasLtMatrixLayout_t desc,
    cublasLtMatrixLayoutAttribute_t attr,
    T* value) {
  size_t written = 0;
  return cublasLtMatrixLayoutGetAttribute(
             desc, attr, value, sizeof(T), &written) ==
      CUBLAS_STATUS_SUCCESS;
}

template <typename T>
bool getMatmulPreferenceAttribute(
    cublasLtMatmulPreference_t desc,
    cublasLtMatmulPreferenceAttributes_t attr,
    T* value) {
  size_t written = 0;
  return cublasLtMatmulPreferenceGetAttribute(
             desc, attr, value, sizeof(T), &written) ==
      CUBLAS_STATUS_SUCCESS;
}

template <typename T>
void appendAttr(std::string& key, const char* name, bool ok, T value) {
  key += c10::str("|", name, "=");
  if (ok) {
    key += c10::str(value);
  } else {
    key += "na";
  }
}

template <typename T, typename F>
void appendMatmulAttr(
    std::string& key,
    const char* name,
    cublasLtMatmulDesc_t desc,
    cublasLtMatmulDescAttributes_t attr,
    T initial,
    F format) {
  T value = initial;
  bool ok = getMatmulDescAttribute(desc, attr, &value);
  appendAttr(key, name, ok, format(value));
}

template <typename T>
void appendMatmulAttr(
    std::string& key,
    const char* name,
    cublasLtMatmulDesc_t desc,
    cublasLtMatmulDescAttributes_t attr,
    T initial) {
  appendMatmulAttr(key, name, desc, attr, initial, [](T value) { return value; });
}

template <typename T, typename F>
void appendLayoutAttr(
    std::string& key,
    const char* name,
    cublasLtMatrixLayout_t desc,
    cublasLtMatrixLayoutAttribute_t attr,
    T initial,
    F format) {
  T value = initial;
  bool ok = getMatrixLayoutAttribute(desc, attr, &value);
  appendAttr(key, name, ok, format(value));
}

template <typename T>
void appendLayoutAttr(
    std::string& key,
    const char* name,
    cublasLtMatrixLayout_t desc,
    cublasLtMatrixLayoutAttribute_t attr,
    T initial) {
  appendLayoutAttr(key, name, desc, attr, initial, [](T value) { return value; });
}

template <typename T>
void appendPreferenceAttr(
    std::string& key,
    const char* name,
    cublasLtMatmulPreference_t desc,
    cublasLtMatmulPreferenceAttributes_t attr,
    T initial) {
  T value = initial;
  bool ok = getMatmulPreferenceAttribute(desc, attr, &value);
  appendAttr(key, name, ok, value);
}

void appendMatmulDescKey(std::string& key, cublasLtMatmulDesc_t desc) {
  auto as_int = [](auto value) { return static_cast<int>(value); };
  auto is_set = [](const void* value) { return value != nullptr; };

  appendMatmulAttr(
      key, "compute", desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, CUBLAS_COMPUTE_32F, as_int);
  appendMatmulAttr(key, "scale", desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, CUDA_R_32F, as_int);
  appendMatmulAttr(key, "transa", desc, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_N, as_int);
  appendMatmulAttr(key, "transb", desc, CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N, as_int);
  appendMatmulAttr(
      key, "pointer_mode", desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_HOST, as_int);
  appendMatmulAttr(
      key, "epilogue", desc, CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_DEFAULT, as_int);
  appendMatmulAttr(key, "bias_type", desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUDA_R_32F, as_int);
  appendMatmulAttr(
      key, "bias_ptr", desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(
      key, "bias_batch_stride", desc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, int64_t{0});
  appendMatmulAttr(
      key, "a_scale_ptr", desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(
      key, "b_scale_ptr", desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(
      key, "d_scale_ptr", desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(
      key, "aux_ptr", desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(key, "aux_ld", desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, int64_t{0});
  appendMatmulAttr(
      key, "aux_batch_stride", desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, int64_t{0});
  appendMatmulAttr(key, "sm", desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, static_cast<int32_t>(0));

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  appendMatmulAttr(
      key, "fast_accum", desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, static_cast<int8_t>(0), as_int);
  appendMatmulAttr(
      key, "c_scale_ptr", desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(
      key, "amax_d_ptr", desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, static_cast<void*>(nullptr), is_set);
  appendMatmulAttr(key, "aux_type", desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, CUDA_R_32F, as_int);
#endif
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  appendMatmulAttr(key, "a_scale_mode", desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, 0);
  appendMatmulAttr(key, "b_scale_mode", desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, 0);
#endif
}

void appendLayoutKey(
    std::string& key,
    const char* name,
    cublasLtMatrixLayout_t desc) {
  auto as_int = [](auto value) { return static_cast<int>(value); };
  key += c10::str("|", name);
  appendLayoutAttr(key, ".type", desc, CUBLASLT_MATRIX_LAYOUT_TYPE, CUDA_R_32F, as_int);
  appendLayoutAttr(key, ".order", desc, CUBLASLT_MATRIX_LAYOUT_ORDER, CUBLASLT_ORDER_COL, as_int);
  appendLayoutAttr(key, ".rows", desc, CUBLASLT_MATRIX_LAYOUT_ROWS, static_cast<uint64_t>(0));
  appendLayoutAttr(key, ".cols", desc, CUBLASLT_MATRIX_LAYOUT_COLS, static_cast<uint64_t>(0));
  appendLayoutAttr(key, ".ld", desc, CUBLASLT_MATRIX_LAYOUT_LD, int64_t{0});
  appendLayoutAttr(key, ".batch", desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, int32_t{0});
  appendLayoutAttr(
      key, ".stride", desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, int64_t{0});
}

void appendPreferenceKey(std::string& key, cublasLtMatmulPreference_t desc) {
  appendPreferenceAttr(
      key, "workspace", desc, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, size_t{0});
  appendPreferenceAttr(
      key, "reduction", desc, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, uint32_t{0});
  appendPreferenceAttr(
      key, "align_a", desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, uint32_t{0});
  appendPreferenceAttr(
      key, "align_b", desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, uint32_t{0});
  appendPreferenceAttr(
      key, "align_c", desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, uint32_t{0});
  appendPreferenceAttr(
      key, "align_d", desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, uint32_t{0});
}

// Mirror of _getAlignment in CUDABlas.cpp:217. Returns the largest power-of-2
// alignment that divides the address, capped at 256.
size_t getPtrAlignment(uintptr_t addr) {
  size_t alignment = 256;
  while (alignment > 1 && (addr % alignment) != 0) {
    alignment /= 2;
  }
  return alignment;
}

std::string cublasLtMatmulCacheKey(
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t adesc,
    cublasLtMatrixLayout_t bdesc,
    cublasLtMatrixLayout_t cdesc,
    cublasLtMatrixLayout_t ddesc,
    cublasLtMatrixLayout_t heuristic_bdesc,
    cublasLtMatrixLayout_t heuristic_cdesc,
    cublasLtMatrixLayout_t heuristic_ddesc,
    cublasLtMatmulPreference_t preference,
    size_t real_d_alignment) {
  auto prop = at::cuda::getCurrentDeviceProperties();
  std::string key = c10::str(
      "device=",
      static_cast<int>(c10::cuda::current_device()),
      "|cc=",
      prop->major,
      ".",
      prop->minor);
  appendMatmulDescKey(key, operation_desc);
  appendLayoutKey(key, "a", adesc);
  appendLayoutKey(key, "b", bdesc);
  appendLayoutKey(key, "c", cdesc);
  appendLayoutKey(key, "d", ddesc);
  appendLayoutKey(key, "heuristic_b", heuristic_bdesc);
  appendLayoutKey(key, "heuristic_c", heuristic_cdesc);
  appendLayoutKey(key, "heuristic_d", heuristic_ddesc);
  appendPreferenceKey(key, preference);
  // The autotune timing loop writes to a scratch buffer whose alignment we
  // match to the real D pointer (see scratch_d below). Two calls with the
  // same descriptors but different real D alignments get separate cache
  // entries because an algo with a 256-byte fast path could be best for one
  // alignment class and not the other.
  key += c10::str("|real_d_align=", real_d_alignment);
  return key;
}

struct CublasLtAlgoConfig {
  int32_t id = 0;
  uint32_t tile = 0;
  uint32_t stages = 0;
  int32_t splitk = 1;
  uint32_t reduction = 0;
  uint32_t swizzle = 0;
  uint32_t custom = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  // Hopper+ thread-block cluster and inner-shape selection. Defaults are
  // CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED and CUBLASLT_CLUSTER_SHAPE_AUTO;
  // without these in the snapshot a Hopper algo with a non-default cluster
  // would reconstruct to a different kernel after load.
  uint16_t inner_shape = 0;
  uint16_t cluster_shape = 0;
#endif
};

template <typename T>
bool getAlgoConfigAttribute(
    const cublasLtMatmulAlgo_t* algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    T* value) {
  size_t written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(
             algo, attr, value, sizeof(T), &written) == CUBLAS_STATUS_SUCCESS;
}

bool algoConfigFromAlgo(
    const cublasLtMatmulAlgo_t& algo,
    CublasLtAlgoConfig* config) {
  bool ok = getAlgoConfigAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &config->id) &&
      getAlgoConfigAttribute(
             &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &config->tile) &&
      getAlgoConfigAttribute(
             &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &config->stages) &&
      getAlgoConfigAttribute(
             &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &config->splitk) &&
      getAlgoConfigAttribute(
             &algo,
             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
             &config->reduction) &&
      getAlgoConfigAttribute(
             &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &config->swizzle) &&
      getAlgoConfigAttribute(
             &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &config->custom);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  ok = ok &&
      getAlgoConfigAttribute(
             &algo,
             CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
             &config->inner_shape) &&
      getAlgoConfigAttribute(
             &algo,
             CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
             &config->cluster_shape);
#endif
  return ok;
}

std::string algoConfigName(const CublasLtAlgoConfig& config) {
  std::string name = c10::str(
      "CublasLtMatmul_id_",
      config.id,
      "_tile_",
      config.tile,
      "_stages_",
      config.stages,
      "_splitk_",
      config.splitk,
      "_red_",
      config.reduction,
      "_swizzle_",
      config.swizzle,
      "_custom_",
      config.custom);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  name += c10::str(
      "_inner_", config.inner_shape, "_cluster_", config.cluster_shape);
#endif
  return name;
}

bool algoConfigFromName(const std::string& name, CublasLtAlgoConfig* config) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  int matched = std::sscanf(
      name.c_str(),
      "CublasLtMatmul_id_%d_tile_%u_stages_%u_splitk_%d_red_%u_swizzle_%u_custom_%u"
      "_inner_%hu_cluster_%hu",
      &config->id,
      &config->tile,
      &config->stages,
      &config->splitk,
      &config->reduction,
      &config->swizzle,
      &config->custom,
      &config->inner_shape,
      &config->cluster_shape);
  return matched == 9;
#else
  int matched = std::sscanf(
      name.c_str(),
      "CublasLtMatmul_id_%d_tile_%u_stages_%u_splitk_%d_red_%u_swizzle_%u_custom_%u",
      &config->id,
      &config->tile,
      &config->stages,
      &config->splitk,
      &config->reduction,
      &config->swizzle,
      &config->custom);
  return matched == 7;
#endif
}

template <typename T>
bool setAlgoConfigAttribute(
    cublasLtMatmulAlgo_t* algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    const T& value) {
  return cublasLtMatmulAlgoConfigSetAttribute(
             algo, attr, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS;
}

bool initializeAlgo(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t adesc,
    cublasLtMatrixLayout_t bdesc,
    cublasLtMatrixLayout_t cdesc,
    cublasLtMatrixLayout_t ddesc,
    const CublasLtAlgoConfig& config,
    cublasLtMatmulAlgo_t* algo) {
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  cudaDataType_t scale_type = CUDA_R_32F;
  cudaDataType_t a_type = CUDA_R_32F;
  cudaDataType_t b_type = CUDA_R_32F;
  cudaDataType_t c_type = CUDA_R_32F;
  cudaDataType_t d_type = CUDA_R_32F;

  if (!getMatmulDescAttribute(
          operation_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &compute_type) ||
      !getMatmulDescAttribute(
          operation_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type) ||
      !getMatrixLayoutAttribute(adesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &a_type) ||
      !getMatrixLayoutAttribute(bdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &b_type) ||
      !getMatrixLayoutAttribute(cdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &c_type) ||
      !getMatrixLayoutAttribute(ddesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &d_type)) {
    return false;
  }

  auto status = cublasLtMatmulAlgoInit(
      handle,
      compute_type,
      scale_type,
      a_type,
      b_type,
      c_type,
      d_type,
      config.id,
      algo);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return false;
  }

  bool ok = setAlgoConfigAttribute(
                algo, CUBLASLT_ALGO_CONFIG_TILE_ID, config.tile) &&
      setAlgoConfigAttribute(
             algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, config.stages) &&
      setAlgoConfigAttribute(
             algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, config.splitk) &&
      setAlgoConfigAttribute(
             algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, config.reduction) &&
      setAlgoConfigAttribute(
             algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, config.swizzle) &&
      setAlgoConfigAttribute(
             algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, config.custom);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  ok = ok &&
      setAlgoConfigAttribute(
             algo,
             CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
             config.inner_shape) &&
      setAlgoConfigAttribute(
             algo,
             CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
             config.cluster_shape);
#endif
  return ok;
}

size_t cudaDataTypeSize(cudaDataType_t type) {
  switch (type) {
    case CUDA_R_8I:
    case CUDA_R_8U:
    case CUDA_C_8I:
    case CUDA_C_8U:
    case CUDA_R_8F_E4M3:
    case CUDA_R_8F_E5M2:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    case CUDA_R_8F_UE8M0:
    case CUDA_R_4F_E2M1:
#endif
      return 1;
    case CUDA_R_16F:
    case CUDA_R_16BF:
    case CUDA_R_16I:
    case CUDA_R_16U:
      return 2;
    case CUDA_R_32F:
    case CUDA_R_32I:
    case CUDA_R_32U:
    case CUDA_C_16F:
    case CUDA_C_16I:
    case CUDA_C_16U:
      return 4;
    case CUDA_R_64F:
    case CUDA_R_64I:
    case CUDA_R_64U:
    case CUDA_C_32F:
    case CUDA_C_32I:
    case CUDA_C_32U:
      return 8;
    case CUDA_C_64F:
    case CUDA_C_64I:
    case CUDA_C_64U:
      return 16;
    default:
      return 0;
  }
}

size_t layoutSizeBytes(cublasLtMatrixLayout_t desc) {
  cudaDataType_t type = CUDA_R_32F;
  cublasLtOrder_t order = CUBLASLT_ORDER_COL;
  uint64_t rows = 0;
  uint64_t cols = 0;
  int64_t ld = 0;
  int32_t batch_count = 0;
  int64_t stride = 0;

  if (!getMatrixLayoutAttribute(desc, CUBLASLT_MATRIX_LAYOUT_TYPE, &type) ||
      !getMatrixLayoutAttribute(desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order) ||
      !getMatrixLayoutAttribute(desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows) ||
      !getMatrixLayoutAttribute(desc, CUBLASLT_MATRIX_LAYOUT_COLS, &cols) ||
      !getMatrixLayoutAttribute(desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld)) {
    return 0;
  }

  getMatrixLayoutAttribute(
      desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count);
  getMatrixLayoutAttribute(
      desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride);
  batch_count = std::max(batch_count, 1);

  size_t matrix_elements = 0;
  if (order == CUBLASLT_ORDER_COL) {
    matrix_elements = static_cast<size_t>(ld) * std::max<uint64_t>(cols, 1);
  } else {
    matrix_elements = static_cast<size_t>(ld) * std::max<uint64_t>(rows, 1);
  }

  size_t elements = matrix_elements * static_cast<size_t>(batch_count);
  if (batch_count > 1 && stride > 0) {
    elements =
        static_cast<size_t>(stride) * static_cast<size_t>(batch_count - 1) +
        matrix_elements;
  }

  return elements * cudaDataTypeSize(type);
}

bool validateCachedAlgo(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t adesc,
    cublasLtMatrixLayout_t bdesc,
    cublasLtMatrixLayout_t cdesc,
    cublasLtMatrixLayout_t ddesc,
    size_t workspace_size,
    const std::string& config_name,
    cublasLtMatmulHeuristicResult_t* result) {
  CublasLtAlgoConfig config;
  cublasLtMatmulAlgo_t algo = {};
  if (!algoConfigFromName(config_name, &config) ||
      !initializeAlgo(
          handle, operation_desc, adesc, bdesc, cdesc, ddesc, config, &algo)) {
    return false;
  }

  cublasLtMatmulHeuristicResult_t check_result = {};
  auto status = cublasLtMatmulAlgoCheck(
      handle,
      operation_desc,
      adesc,
      bdesc,
      cdesc,
      ddesc,
      &algo,
      &check_result);
  if (status != CUBLAS_STATUS_SUCCESS ||
      check_result.workspaceSize > workspace_size) {
    return false;
  }

  check_result.algo = algo;
  *result = check_result;
  return true;
}

} // anonymous namespace

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
    int* returned_result) {
  auto get_top_heuristic = [&]() {
    return cublasLtMatmulAlgoGetHeuristic(
        handle,
        operation_desc,
        adesc,
        heuristic_bdesc,
        heuristic_cdesc,
        heuristic_ddesc,
        preference,
        1,
        heuristic_result,
        returned_result);
  };

  auto* ctx = getTuningContext();
  if (!ctx->IsTunableOpEnabled()) {
    return get_top_heuristic();
  }

  constexpr const char* op_signature = "CublasLtMatmulHeuristic";
  const size_t real_d_alignment =
      getPtrAlignment(reinterpret_cast<uintptr_t>(d));
  const std::string params_signature = cublasLtMatmulCacheKey(
      operation_desc,
      adesc,
      bdesc,
      cdesc,
      ddesc,
      heuristic_bdesc,
      heuristic_cdesc,
      heuristic_ddesc,
      preference,
      real_d_alignment);
  auto& manager = ctx->GetTuningResultsManager();
  auto result = manager.Lookup(op_signature, params_signature);
  if (result != ResultEntry::Null() && result != ResultEntry::Default()) {
    if (validateCachedAlgo(
            handle,
            operation_desc,
            adesc,
            bdesc,
            cdesc,
            ddesc,
            workspace_size,
            result.GetKey(),
            heuristic_result)) {
      *returned_result = 1;
      return CUBLAS_STATUS_SUCCESS;
    }
    manager.Delete(op_signature, params_signature);
  } else if (result == ResultEntry::Default()) {
    return get_top_heuristic();
  }

  int requested_algo_count = ctx->GetCublasLtRequestedAlgoCount();
  if (ctx->IsTuningEnabled() && requested_algo_count == 1) {
    TORCH_WARN_ONCE(
        "cuBLASLt autotuning is enabled, but "
        "the requested cuBLASLt heuristic algorithm count is 1. "
        "Autotuning will not run; using the top heuristic result. "
        "Set a value greater than 1 with "
        "torch.cuda.tunable.set_cublaslt_requested_algo_count(count) or "
        "PYTORCH_TUNABLEOP_CUBLASLT_REQUESTED_ALGO_COUNT.");
    return get_top_heuristic();
  }

  if (!ctx->IsTuningEnabled()) {
    TORCH_WARN_ONCE(
        "TunableOp is enabled, but tuning is disabled and no cached "
        "cuBLASLt algorithm was found. Falling back to the top heuristic "
        "result.");
    return get_top_heuristic();
  }

  bool is_capturing =
      c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
      c10::cuda::CaptureStatus::None;
  if (is_capturing) {
    TORCH_WARN_ONCE(
        "CUDA graph capture is in progress and cuBLASLt autotuning cannot "
        "happen during capture. Falling back to the top heuristic result.");
    return get_top_heuristic();
  }

  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
      requested_algo_count);
  int returned_results = 0;
  auto status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      operation_desc,
      adesc,
      heuristic_bdesc,
      heuristic_cdesc,
      heuristic_ddesc,
      preference,
      requested_algo_count,
      heuristic_results.data(),
      &returned_results);
  if (status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
    *returned_result = returned_results;
    return status;
  }

  // Scratch is offset within its allocation so its alignment matches the user's real D pointer
  constexpr size_t kScratchAlignmentPadding = 128;
  size_t d_size = layoutSizeBytes(ddesc);
  // d_size == 0 means a layout attribute query failed; benchmarking with an
  // undersized scratch would let cublasLtMatmul write past it (no bounds check).
  if (d_size == 0) {
    *heuristic_result = heuristic_results[0];
    *returned_result = 1;
    return CUBLAS_STATUS_SUCCESS;
  }
  c10::DataPtr scratch;
  try {
    scratch = c10::cuda::CUDACachingAllocator::get()->allocate(
        d_size + kScratchAlignmentPadding);
  } catch (const c10::Error&) {
    TORCH_WARN_ONCE(
        "cuBLASLt autotuning could not allocate scratch output space to "
        "benchmark heuristic algorithms. Falling back to the top heuristic "
        "result.");
    *heuristic_result = heuristic_results[0];
    *returned_result = 1;
    return CUBLAS_STATUS_SUCCESS;
  }
  const size_t alignment_offset =
      (real_d_alignment == 256) ? 0 : real_d_alignment;
  void* scratch_d = static_cast<char*>(scratch.get()) + alignment_offset;

  CudaEvent start_event;
  CudaEvent stop_event;

  int best_algo_idx = -1;
  float best_algo_time = 0.0f;
  std::vector<float> algo_times(kCublasLtAutotuneRepeats);
  for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
    if (heuristic_results[algo_idx].state != CUBLAS_STATUS_SUCCESS ||
        heuristic_results[algo_idx].workspaceSize > workspace_size) {
      continue;
    }

    auto run_matmul = [&]() {
      return cublasLtMatmul(
          handle,
          operation_desc,
          alpha,
          a,
          adesc,
          b,
          bdesc,
          beta,
          c,
          cdesc,
          scratch_d,
          ddesc,
          &heuristic_results[algo_idx].algo,
          workspace,
          heuristic_results[algo_idx].workspaceSize,
          stream);
    };

    status = run_matmul();
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }

    bool measured = true;
    for (int check_idx = 0; check_idx < kCublasLtAutotuneRepeats;
         ++check_idx) {
      AT_CUDA_CHECK(cudaEventRecord(start_event, stream));
      status = run_matmul();
      if (status != CUBLAS_STATUS_SUCCESS) {
        measured = false;
        break;
      }
      AT_CUDA_CHECK(cudaEventRecord(stop_event, stream));
      AT_CUDA_CHECK(cudaEventSynchronize(stop_event));
      AT_CUDA_CHECK(cudaEventElapsedTime(
          &algo_times[check_idx], start_event, stop_event));
    }

    if (!measured) {
      continue;
    }

    std::sort(algo_times.begin(), algo_times.end());
    float time = algo_times[algo_times.size() / 2];
    if (algo_times.size() % 2 == 0) {
      time = (time + algo_times[algo_times.size() / 2 - 1]) / 2.0f;
    }

    if (best_algo_idx < 0 || time < best_algo_time) {
      best_algo_idx = algo_idx;
      best_algo_time = time;
    }
  }

  if (best_algo_idx < 0) {
    // If all candidates fail measurement runs, surface the last status directly
    *returned_result = 0;
    return status;
  }

  CublasLtAlgoConfig config;
  if (algoConfigFromAlgo(heuristic_results[best_algo_idx].algo, &config)) {
    manager.Add(
        op_signature,
        params_signature,
        ResultEntry(algoConfigName(config), best_algo_time));
  }

  *heuristic_result = heuristic_results[best_algo_idx];
  *returned_result = 1;
  return CUBLAS_STATUS_SUCCESS;
}

} // namespace at::cuda::tunable

#endif // USE_ROCM
