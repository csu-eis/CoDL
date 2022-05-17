
#include "mace/ops/common/gemmlowp_util.h"
#include "mace/ops/matmul.h"
#include "test/codl_run/utils/matmul_util.h"

namespace mace {

void MatMulUtils::Validate(const Tensor *A,
                           const Tensor *B,
                           const bool transpose_a,
                           const bool transpose_b) {
  const index_t lhs_rank = A->dim_size();
  const index_t rhs_rank = B->dim_size();

#if 0
  LOG(INFO) << "A_shape " << VectorToString<index_t>(A->shape())
            << ", A_image_shape " << VectorToString<size_t>(A->image_shape())
            << ", B_shape " << VectorToString<index_t>(B->shape())
            << ", B_image_shape " << VectorToString<size_t>(B->image_shape())
            << ", transpose_a " << transpose_a
            << ", transpose_b " << transpose_b;
#endif

  MACE_CHECK(lhs_rank >= 2 && rhs_rank >= 2,
             "rank should be greater than or equal to 2");
  if (lhs_rank == rhs_rank) {
    for (index_t i = 0; i < A->dim_size() - 2; ++i) {
      MACE_CHECK(A->dim(i) == B->dim(i),
                 "batch dimensions are not equal: ",
                 A->dim(i),
                 " vs. ",
                 B->dim(i));
    }
  } else {
    MACE_CHECK(lhs_rank == 2 || rhs_rank == 2,
               "Either lhs or rhs matrix should has rank 2 "
               "for non-batched matrix multiplication");
  }

  index_t
      lhs_depth = transpose_a ? A->dim(lhs_rank - 2) : A->dim(lhs_rank - 1);
  index_t
      rhs_depth = transpose_b ? B->dim(rhs_rank - 1) : B->dim(rhs_rank - 2);
  MACE_CHECK(lhs_depth == rhs_depth, "the number of A's column ", lhs_depth,
             " must be equal to B's row ", rhs_depth);
}

MaceStatus MatMulCpuFloatKernel::Compute(
    OpContext *context,
    const Tensor *lhs,
    const Tensor *rhs,
    const Tensor *bias,
    const bool transpose_a,
    const bool transpose_b,
    Tensor *C) {
  MatMulUtils::Validate(lhs, rhs, transpose_a, transpose_b);
  const index_t lhs_rank = lhs->dim_size();
  const index_t lhs_rows = lhs->dim(lhs_rank - 2);
  const index_t lhs_cols = lhs->dim(lhs_rank - 1);
  const index_t rhs_rank = rhs->dim_size();
  const index_t rhs_rows = rhs->dim(rhs_rank - 2);
  const index_t rhs_cols = rhs->dim(rhs_rank - 1);

  const index_t rows = transpose_a ? lhs_cols : lhs_rows;
  const index_t cols = transpose_b ? rhs_rows : rhs_cols;
  const index_t depth = transpose_a ? lhs_rows : lhs_cols;
  const index_t
      lhs_batch =
      std::accumulate(lhs->shape().begin(), lhs->shape().end() - 2, 1,
                      std::multiplies<index_t>());
  const index_t
      rhs_batch =
      std::accumulate(rhs->shape().begin(), rhs->shape().end() - 2, 1,
                      std::multiplies<index_t>());
  index_t batch = 1;
  std::vector<index_t> output_shape;
  if (lhs_rank >= rhs_rank) {
    output_shape = lhs->shape();
    output_shape[lhs_rank - 2] = rows;
    output_shape[lhs_rank - 1] = cols;
    batch = lhs_batch;
  } else {
    output_shape = rhs->shape();
    output_shape[rhs_rank - 2] = rows;
    output_shape[rhs_rank - 1] = cols;
    batch = rhs_batch;
  }
  bool lhs_batched = true;
  bool rhs_batched = true;
  if (lhs_rank < rhs_rank) {
    lhs_batched = false;
  } else if (rhs_rank < lhs_rank) {
    rhs_batched = false;
  }

  MACE_RETURN_IF_ERROR(C->Resize(output_shape));

#if 0
  LOG(INFO) << "lhs " << VectorToString(lhs->shape())
            << ", rhs " << VectorToString(rhs->shape())
            << ", C " << VectorToString(C->shape());
#endif

  if (rows == 1 && transpose_b) {
    return gemv_.Compute(context,
                         rhs,
                         lhs,
                         bias,
                         batch,
                         cols,
                         depth,
                         rhs_batched,
                         lhs_batched,
                         C);
  } else if (cols == 1 && !transpose_a) {
    return gemv_.Compute(context,
                         lhs,
                         rhs,
                         bias,
                         batch,
                         rows,
                         depth,
                         lhs_batched,
                         rhs_batched,
                         C);
  } else {
    context->cpu_device()->scratch_buffer()->Rewind();
    MaceStatus ret = gemm_.Compute(context,
                                   lhs,
                                   rhs,
                                   batch,
                                   lhs_rows,
                                   lhs_cols,
                                   rhs_rows,
                                   rhs_cols,
                                   transpose_a,
                                   transpose_b,
                                   false,
                                   lhs_batched,
                                   rhs_batched,
                                   C);
    if (bias != nullptr) {
      MACE_CHECK(bias->dim_size() == 1 && bias->dim(0) == cols,
                 "bias' dim should be <= 2.");
      Tensor::MappingGuard bias_guard(bias);
      Tensor::MappingGuard c_guard(C);
      const float *bias_data = bias->data<float>();
      float *c_data = C->mutable_data<float>();

      utils::ThreadPool
          &thread_pool = context->device()->cpu_runtime()->thread_pool();

      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t w = start1; w < end1; w += step1) {
            c_data[i * cols + w] += bias_data[w];
          }
        }
      }, 0, batch * rows, 1, 0, cols, 1);
    }

    return ret;
  }
}

#ifdef MACE_ENABLE_QUANTIZE

MaceStatus MatMulCpuUint8Kernel::Compute(
    OpContext *context,
    const Tensor *lhs,
    const Tensor *rhs,
    const Tensor *bias,
    const bool transpose_a,
    const bool transpose_b,
    Tensor *C) {
  MACE_UNUSED(bias);
  MatMulUtils::Validate(lhs, rhs, transpose_a, transpose_b);
  const index_t lhs_rank = lhs->dim_size();
  const index_t lhs_rows = lhs->dim(lhs_rank - 2);
  const index_t lhs_cols = lhs->dim(lhs_rank - 1);
  const index_t rhs_rank = rhs->dim_size();
  const index_t rhs_rows = rhs->dim(rhs_rank - 2);
  const index_t rhs_cols = rhs->dim(rhs_rank - 1);

  const index_t rows = transpose_a ? lhs_cols : lhs_rows;
  const index_t cols = transpose_b ? rhs_rows : rhs_cols;
  const index_t depth = transpose_a ? lhs_rows : lhs_cols;
  const index_t
      lhs_batch =
      std::accumulate(lhs->shape().begin(), lhs->shape().end() - 2, 1,
                      std::multiplies<index_t>());
  const index_t
      rhs_batch =
      std::accumulate(rhs->shape().begin(), rhs->shape().end() - 2, 1,
                      std::multiplies<index_t>());
  index_t batch = 1;
  std::vector<index_t> output_shape;
  if (lhs_rank >= rhs_rank) {
    output_shape = lhs->shape();
    output_shape[lhs_rank - 2] = rows;
    output_shape[lhs_rank - 1] = cols;
    batch = lhs_batch;
  } else {
    output_shape = rhs->shape();
    output_shape[rhs_rank - 2] = rows;
    output_shape[rhs_rank - 1] = cols;
    batch = rhs_batch;
  }
  bool lhs_batched = true;
  bool rhs_batched = true;
  if (lhs_rank < rhs_rank) {
    lhs_batched = false;
  } else if (rhs_rank < lhs_rank) {
    rhs_batched = false;
  }

  MACE_RETURN_IF_ERROR(C->Resize(output_shape));

  constexpr gemmlowp::MapOrder kRowMajor = gemmlowp::MapOrder::RowMajor;
  constexpr gemmlowp::MapOrder kColMajor = gemmlowp::MapOrder::ColMajor;

#define MATMUL_FIXPOINT_IMPL(AOrder, BOrder, OutType)           \
  ops::MatMulFixpointImpl<AOrder, BOrder, OutType>()(           \
      context, lhs, rhs, batch, rows, depth, cols, lhs_batched, rhs_batched, C);

#define MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT(OutType)        \
  if (transpose_a) {                                          \
    if (transpose_b) {                                        \
      MATMUL_FIXPOINT_IMPL(kColMajor, kColMajor, OutType);    \
    } else {                                                  \
      MATMUL_FIXPOINT_IMPL(kColMajor, kRowMajor, OutType);    \
    }                                                         \
  } else {                                                    \
    if (transpose_b) {                                        \
      MATMUL_FIXPOINT_IMPL(kRowMajor, kColMajor, OutType);    \
    } else {                                                  \
      MATMUL_FIXPOINT_IMPL(kRowMajor, kRowMajor, OutType);    \
    }                                                         \
  }

  MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT(uint8_t);

#undef MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT
#undef MATMUL_FIXPOINT_IMPL

  return MaceStatus::MACE_SUCCESS;
}

#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
std::unique_ptr<ops::OpenCLMatMulKernel>
    CreateOpenCLMatMulKernel(const MemoryType mtype) {
  switch (mtype) {
    case MemoryType::GPU_IMAGE:
      return std::unique_ptr<ops::OpenCLMatMulKernel>(
          new ops::opencl::image::MatMulKernel());
    case MemoryType::GPU_BUFFER:
      return std::unique_ptr<ops::OpenCLMatMulKernel>(
          new ops::opencl::buffer::MatMulKernel());
    default:
      LOG(ERROR) << "Not support memory type";
      return nullptr;
  }
}
#endif

}  // namespace mace
