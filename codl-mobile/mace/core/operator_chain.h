
#ifndef MACE_CORE_OPERATOR_CHAIN_H_
#define MACE_CORE_OPERATOR_CHAIN_H_

#include <vector>
#include <string>
#include "mace/core/operator.h"

namespace mace {

class OperatorChain {
 public:
  OperatorChain() {}

  OperatorChain(Operation *op) {
    push_back(op);
  }

  inline size_t size() const {
    return operators_.size();
  }

  inline void push_back(Operation *op) {
    if (operators_.size() > 0) {
      if (operators_.size() > 1) {
        operators_.back()->chain_context()->set_position(OP_POSITION_MIDDLE);
      }
      op->chain_context()->set_position(OP_POSITION_TAIL);
    } else if (operators_.size() == 0) {
      op->chain_context()->set_position(OP_POSITION_HEAD);
    }
    
    operators_.push_back(op);
    op->chain_context()->set_chain(reinterpret_cast<void *>(this));
  }

  inline Operation *back() {
    return operators_.back();
  }

  inline Operation *op(const size_t i) {
    MACE_CHECK(i < operators_.size());
    return operators_[i];
  }

  std::string DebugInfo() const;

  MaceStatus Run(OpContext *context);

 private:
  std::vector<Operation *> operators_;
};

}  // namespace mace

#endif  // MACE_CORE_OPERATOR_CHAIN_H_
