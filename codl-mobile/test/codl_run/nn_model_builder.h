
#ifndef TEST_CODL_RUN_NN_MODEL_BUILDER_H_
#define TEST_CODL_RUN_NN_MODEL_BUILDER_H_

#include <vector>
#include "test/codl_run/op_test_task_chain.h"

class NnModelBuilder {
 public:
  NnModelBuilder(const size_t op_count,
                 const size_t op_chain_count,
                 const CodlOpChainCommonParam &common_param)
      : op_count_(op_count),
        op_chain_count_(op_chain_count),
        common_param_(common_param) {}

  virtual ~NnModelBuilder() = default;

  inline size_t op_count() const {
    return op_count_;
  }

  inline size_t op_chain_count() const {
    return op_chain_count_;
  }

  virtual void Build(
      const int chain_idx,
      const std::vector<int> chain_lengths,
      const std::vector<int> pdims,
      const std::vector<float> pratioes,
      std::vector<std::shared_ptr<CodlOpChainParam>> &params) = 0;

  void Build(const int chain_idx,
             const int pdim,
             const float pratio,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
    const std::vector<int> pdims(op_count_, pdim);
    const std::vector<float> pratioes(op_count_, pratio);
    const std::vector<int> chain_lengths;
    Build(chain_idx, chain_lengths, pdims, pratioes, params);
  }

  void Build(const int pdim,
             const float pratio,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) {
    Build(-1, pdim, pratio, params);
  }

  virtual void Build(
      const int chain_idx,
      const int chain_param_hint,
      std::vector<std::shared_ptr<CodlOpChainParam>> &params) = 0;
  
 protected:
  size_t op_count_;
  size_t op_chain_count_;
  CodlOpChainCommonParam common_param_;
};

class Yolov2Builder : public NnModelBuilder {
 public:
  Yolov2Builder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(27, 27, common_param) {}

  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class PosenetBuilder : public NnModelBuilder {
 public:
  PosenetBuilder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(34, 34, common_param) {}
  
  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class AlexnetBuilder : public NnModelBuilder {
 public:
  AlexnetBuilder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(11, 11, common_param) {}
  
  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class Vgg16Builder : public NnModelBuilder {
 public:
  Vgg16Builder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(21, 21, common_param) {}
  
  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class FastStyleTransferBuilder : public NnModelBuilder {
 public:
  FastStyleTransferBuilder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(14, 14, common_param) {}

  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class Resnet50v1Builder : public NnModelBuilder {
 public:
  Resnet50v1Builder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(51, 51, common_param) {}

  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class RetinaFaceBuilder : public NnModelBuilder {
 public:
  RetinaFaceBuilder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(94 - 7, 94 - 7, common_param) {}

  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

class MobileNetV1Builder : public NnModelBuilder {
 public:
  MobileNetV1Builder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(16, 16, common_param) {}

  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

struct BertConfig {
  index_t L;
  index_t H;
  index_t A;
};

class BertBuilder : public NnModelBuilder {
 public:
  BertBuilder(const CodlOpChainCommonParam &common_param,
              const index_t L,
              const index_t H,
              const index_t A)
      : NnModelBuilder(9, 9, common_param) {
    config_ = {L, H, A};
  }

  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
 private:
  BertConfig config_;
};

class OpChainNetBuilder : public NnModelBuilder {
 public:
  OpChainNetBuilder(const CodlOpChainCommonParam &common_param)
      : NnModelBuilder(30, 30, common_param) {}
  
  void Build(const int chain_idx,
             const std::vector<int> chain_lengths,
             const std::vector<int> pdims,
             const std::vector<float> pratioes,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;

  void Build(const int chain_idx,
             const int chain_param_hint,
             std::vector<std::shared_ptr<CodlOpChainParam>> &params) override;
};

#endif  // TEST_CODL_RUN_NN_MODEL_BUILDER_H_
