
#ifndef MACE_CORE_LATENCY_PREDICTOR_H_
#define MACE_CORE_LATENCY_PREDICTOR_H_

#include <vector>
#include "mace/core/op_context.h"

namespace mace {

enum LPBackend {
  CONCURRENCY_AWARE_BACKEND = 0,
  FLOPS_BACKEND = 1
};

class LPInitContext {
 public:
  LPInitContext() : backend_(CONCURRENCY_AWARE_BACKEND), cur_idx_(0) {}

  inline LPBackend backend() const {
    return backend_;
  }

  inline void set_backend(const LPBackend backend) {
    backend_ = backend;
  }

  inline index_t cur_idx() const {
    return cur_idx_;
  }

  inline void set_cur_idx(const index_t idx) {
    cur_idx_ = idx;
  }
 
 private:
  LPBackend backend_;
  index_t cur_idx_;
};

class LatencyPredictor {
 public:
  LatencyPredictor() : model_count_(0) {}

  LatencyPredictor(const index_t model_count) : model_count_(model_count) {}

  virtual ~LatencyPredictor() {}

  virtual MaceStatus Init(LPInitContext *context) = 0;

  virtual double Predict(OpContext *context,
                         const std::vector<double> &inputs) = 0;
 
 protected:
  index_t model_count_;
};

}  // namespace mace

#endif  // MACE_CORE_LATENCY_PREDICTOR_H_
