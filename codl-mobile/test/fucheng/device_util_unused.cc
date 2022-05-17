
int TestDeviceContext::InitGpuDevice() {

  if (thread_pool_) {
    // Normal pointer.
    //thread_pool = new utils::ThreadPool(1, affinity_policy_);
    //thread_pool->Init();
    // Unique pointer.
    //gpu_thread_pool_.reset(new utils::ThreadPool(1, affinity_policy_));
    //gpu_thread_pool_->Init();
    //thread_pool = gpu_thread_pool_.get();
  }
}
