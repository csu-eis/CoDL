
#include "mace/core/operator_chain.h"
#include "mace/core/co_operator.h"
#include "mace/utils/thread_pool.h"

namespace mace {

std::string OperatorChain::DebugInfo() const {
  std::stringstream stream;
  stream << "[";
  for (auto &op : operators_) {
    stream << "<" << static_cast<int>(op->chain_context()->position())
           << "," << op->debug_def().output(0) << ">";
    if (op != operators_.back()) {
      stream << ",";
    }
  }
  stream << "]";
  return stream.str();
}

MaceStatus OperatorChain::Run(OpContext *context) {
  LOG(INFO) << "Running chain " << DebugInfo();
  utils::ThreadPool::Sleep(1000);

  std::vector<CoOperation *> co_ops;
  StatsFuture map_future;
  cl::UserEvent *unmap_event = nullptr;
  OpenCLEventManager *event_manager
      = context->device()->gpu_runtime()->opencl_runtime()->event_manager();

  for (auto &op : operators_) {
    MACE_CHECK(op->chain_context()->position() != OP_POSITION_NONE);
    co_ops.push_back(reinterpret_cast<CoOperation *>(op));
  }

  CoOperation *front_op = co_ops.front();
  CoOperation *back_op = co_ops.back();

  LOG(INFO) << "Make partitioin and prepare tensors";
  //utils::ThreadPool::Sleep(1000);
  for (auto &op : co_ops) {
    op->MakePartitionPlan(context);
    op->PrepareTemporaryTensors(context);
  }

  LOG(INFO) << "Enqueue premapping";
  //utils::ThreadPool::Sleep(1000);
  for (size_t i = 1; i < co_ops.size(); i ++) {
    co_ops[i]->EnqueueMap(context, &map_future);
  }

  LOG(INFO) << "Enqueue input data transforming and mapping";
  //utils::ThreadPool::Sleep(1000);
  front_op->EnqueueInputDataTransform(context);
  front_op->EnqueueMap(context, &map_future);

  LOG(INFO) << "Enqueue gpu computing";
  //utils::ThreadPool::Sleep(1000);
  for (auto &op : co_ops) {
    op->EnqueueGpuCompute(context);
  }

  LOG(INFO) << "Enqueue output data unmapping and data transforming";
  //utils::ThreadPool::Sleep(1000);
  back_op->EnqueueUnmap(context, &unmap_event);
  back_op->EnqueueOutputDataTransform(context);

  LOG(INFO) << "Enqueue postunmapping";
  //utils::ThreadPool::Sleep(1000);
  for (size_t i = 0; i < co_ops.size() - 1; i ++) {
    co_ops[i]->EnqueueUnmap(context);
  }

  LOG(INFO) << "Map";
  //utils::ThreadPool::Sleep(1000);
  map_future.wait_fn(nullptr);
  
  LOG(INFO) << "Run cpu computing";
  //utils::ThreadPool::Sleep(1000);
  for (auto &op : co_ops) {
    op->RunCpuCompute(context);
  }
  
  LOG(INFO) << "Unmap";
  //utils::ThreadPool::Sleep(1000);
  if (unmap_event != nullptr) {
    event_manager->SetUserEventComplete(unmap_event);
  }

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace
