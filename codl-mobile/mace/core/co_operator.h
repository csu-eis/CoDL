
#ifndef MACE_OPS_CO_OPERATOR_H_
#define MACE_OPS_CO_OPERATOR_H_

#include "mace/core/operator.h"

namespace mace {

enum CpuBufferIdx {
  BUF_IDX_WEIGHTS = 0,
  BUF_IDX_IN      = 1,
  BUF_IDX_OUT     = 2
};

class CoOperation : public Operation {
 public:
  explicit CoOperation(OpConstructContext *context)
      : Operation(context),
        is_first_time_run_(false),
        mem_type_(MemoryType::CPU_BUFFER) {}

  MaceStatus Run(OpContext *context) override {
    if (mem_type_ == MemoryType::GPU_IMAGE) {
      return RunCpuGpuImage(context);
    } else {
      return RunCpuGpuBuffer(context);
    }
  }

  virtual MaceStatus MakePartitionPlan(OpContext *context) {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus PrepareTemporaryTensors(OpContext *context) {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus EnqueueInputDataTransform(
      OpContext *context, StatsFuture *future = nullptr) {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus EnqueueMap(
      OpContext *context, StatsFuture *future = nullptr) {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus EnqueueGpuCompute(
      OpContext *context, StatsFuture *future = nullptr) {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus EnqueueUnmap(OpContext *context,
                                  cl::UserEvent **event = nullptr,
                                  StatsFuture *future = nullptr) {
    MACE_UNUSED(context);
    MACE_UNUSED(event);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus EnqueueOutputDataTransform(
      OpContext *context, StatsFuture *future = nullptr) {
    MACE_UNUSED(context);
    MACE_UNUSED(future);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus RunCpuCompute(OpContext *context) {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  virtual void InitCpu(OpConstructContext *context) {
    MACE_UNUSED(context);
  }

  virtual void InitGpu(OpConstructContext *context) {
    MACE_UNUSED(context);
  }

  virtual MaceStatus TransformWeightGpuToCpu(OpContext *context,
                                             const Tensor *src,
                                             Tensor *dst) {
    MACE_UNUSED(context);
    MACE_UNUSED(src);
    MACE_UNUSED(dst);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus TransformBiasGpuToCpu(OpContext *context,
                                           const Tensor *src,
                                           Tensor *dst) {
    MACE_UNUSED(context);
    MACE_UNUSED(src);
    MACE_UNUSED(dst);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus RunCpuGpuImage(OpContext *context) = 0;
  
  virtual MaceStatus RunCpuGpuImageV2(OpContext *context) {
    MACE_UNUSED(context);
    return MaceStatus::MACE_SUCCESS;
  }

  virtual MaceStatus RunCpuGpuBuffer(OpContext *context) = 0;

 protected:
  bool is_first_time_run_;
  MemoryType mem_type_;
};

}  // namespace mace

#endif  // MACE_OPS_CO_OPERATOR_H_
