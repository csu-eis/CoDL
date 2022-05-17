
#include "mace/ops/fully_connected.h"

namespace mace {
namespace ops {

MaceStatus FullyConnectedGpuFloatOp::TransformWeightGpuToCpu(
    OpContext *context,
    const Tensor *src,
    Tensor *dst) {
  OpenCLBufferType buffer_type = mem_type_ == MemoryType::GPU_IMAGE ?
                                 OpenCLBufferType::WEIGHT_WIDTH :
                                 OpenCLBufferType::CONV2D_FILTER;
  OpenCLBufferTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::CPU_BUFFER)
      .Transform(context, src, buffer_type, MemoryType::CPU_BUFFER, 0, dst);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus FullyConnectedGpuFloatOp::TransformBiasGpuToCpu(
    OpContext *context,
    const Tensor *src,
    Tensor *dst) {
  OpenCLBufferTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::CPU_BUFFER)
      .Transform(context, src, OpenCLBufferType::ARGUMENT,
                 MemoryType::CPU_BUFFER, 0, dst);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus FullyConnectedGpuFloatOp::MakePartitionPlan(
    OpContext *context,
    const Tensor *input,
    const Tensor *weight,
    PartitionDim partition_dim,
    float partition_ratio) {
  //ShowInfo("Make partition plan");
  const DataFormat df = DataFormat::NCHW;
  PartitionResult partition_result;
  if (part_plan_) {
    MACE_CHECK(context->op_runtime_mode() == OpRuntimeMode::RUNNING);
    if (part_plan_->CheckPlanChanged(partition_dim, partition_ratio)) {
      part_plan_.reset(new FullyConnectedPartPlan(partition_dim, partition_ratio, df));
      do {
        partition_result = part_plan_->Make(input->shape(), weight->shape());
      } while (partition_result == PartitionResult::PARTITION_REDO);
      part_plan_->Show();
    }
  } else {
    part_plan_.reset(new FullyConnectedPartPlan(partition_dim, partition_ratio, df));
    do {
      partition_result = part_plan_->Make(input->shape(), weight->shape());
    } while (partition_result == PartitionResult::PARTITION_REDO);
    part_plan_->Show();
  }

  return MaceStatus::MACE_SUCCESS;
}

void RegisterFullyConnected(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "FullyConnected",
                   FullyConnectedOp, DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

  MACE_REGISTER_GPU_OP(op_registry, "FullyConnected", FullyConnectedOp);
}

}  // namespace ops
}  // namespace mace
