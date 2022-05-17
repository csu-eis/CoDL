// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/ops/deconv_2d.h"

namespace mace {
namespace ops {

MaceStatus Deconv2dGpuFloatOp::TransformWeightGpuToCpu(
    OpContext *context,
    const Tensor *src,
    Tensor *dst) {
  OpenCLBufferTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::CPU_BUFFER)
      .Transform(context, src, OpenCLBufferType::CONV2D_FILTER,
                 MemoryType::CPU_BUFFER, 0, dst);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Deconv2dGpuFloatOp::TransformBiasGpuToCpu(
    OpContext *context,
    const Tensor *src,
    Tensor *dst) {
  OpenCLBufferTransformer(MemoryType::GPU_IMAGE,
                          MemoryType::CPU_BUFFER)
      .Transform(context, src, OpenCLBufferType::ARGUMENT,
                 MemoryType::CPU_BUFFER, 0, dst);
  
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Deconv2dGpuFloatOp::MakePartitionPlan(
    OpContext *context,
    const Tensor *output_shape_tensor,
    const Tensor *input,
    const Tensor *filter,
    PartitionDim partition_dim,
    float partition_ratio) {
  //ShowInfo("Make partition plan");
  const DataFormat df = DataFormat::NCHW;
  PartitionResult partition_result;
  if (part_plan_) {
    MACE_CHECK(context->op_runtime_mode() == OpRuntimeMode::RUNNING);
    if (part_plan_->CheckPlanChanged(partition_dim, partition_ratio)) {
      part_plan_.reset(new Deconv2dPartPlan(partition_dim, partition_ratio, df));
      do {
        partition_result = part_plan_->Make(output_shape_tensor,
                                            input->shape(),
                                            filter->shape(),
                                            strides_,
                                            padding_type_,
                                            paddings_,
                                            model_type_);
      } while (partition_result == PartitionResult::PARTITION_REDO);
      part_plan_->Show();
    }
  } else {
    part_plan_.reset(new Deconv2dPartPlan(partition_dim, partition_ratio, df));
    do {
      partition_result = part_plan_->Make(output_shape_tensor,
                                          input->shape(),
                                          filter->shape(),
                                          strides_,
                                          padding_type_,
                                          paddings_,
                                          model_type_);
    } while (partition_result == PartitionResult::PARTITION_REDO);
    part_plan_->Show();
  }

  return MaceStatus::MACE_SUCCESS;
}

void RegisterDeconv2D(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "Deconv2D", Deconv2dOp);
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Deconv2D")
          .SetInputMemoryTypeSetter(
              [](OpConditionContext *context) -> void {
                MemoryType mem_type = MemoryType::CPU_BUFFER;
                if (context->device()->device_type() == DeviceType::GPU) {
                  if (context->device()->gpu_runtime()->UseImageMemory()) {
                    mem_type = MemoryType::GPU_IMAGE;
                  } else {
                    // NOTE(fucheng): Support buffer type.
                    //MACE_NOT_IMPLEMENTED;
                    mem_type = MemoryType::GPU_BUFFER;
                  }
                  context->set_output_mem_type(mem_type);
                  FrameworkType framework_type =
                      static_cast<FrameworkType>(
                        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                            *(context->operator_def()), "framework_type",
                            FrameworkType::TENSORFLOW));
                  if (framework_type == FrameworkType::TENSORFLOW) {
                    context->SetInputInfo(2, MemoryType::CPU_BUFFER,
                                          DataType::DT_INT32);
                  }
                } else {
                  context->set_output_mem_type(mem_type);
                }
              }));
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
