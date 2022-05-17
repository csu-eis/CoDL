
#ifndef TEST_CODL_RUN_OP_TEST_TASK_CHAIN_H_
#define TEST_CODL_RUN_OP_TEST_TASK_CHAIN_H_

#include <vector>
#include <memory>

#include "test/codl_run/core/test_task.h"

namespace mace {

enum CodlOpType {
  CODL_OP_TYPE_NONE,
  CODL_OP_TYPE_CONV2D,
  CODL_OP_TYPE_POOLING,
  CODL_OP_TYPE_FULLY_CONNECTED,
  CODL_OP_TYPE_DECONV2D,
  CODL_OP_TYPE_MATMUL
};

std::string CodlOpTypeToString(const CodlOpType type);

struct CodlOpChainCommonParam {
  bool do_data_transform;
  bool do_compute;
  int num_threads;
  DataType cpu_dtype;
  DataType gpu_dtype;
  MemoryType gpu_mtype;
};

std::string CodlOpChainCommonParamToString(const CodlOpChainCommonParam &param);

class CodlOpChainParam {
 public:
  CodlOpChainParam() : op_type_(CODL_OP_TYPE_NONE),
                       part_dim_(1),
                       part_ratio_(1.0) {}

  CodlOpChainParam(const int part_dim,
                   const float part_ratio,
                   const CodlOpChainCommonParam &param)
      : op_type_(CODL_OP_TYPE_NONE) {
    part_dim_ = part_dim;
    part_ratio_ = part_ratio;
    common_param_ = param;
  }

  virtual ~CodlOpChainParam() noexcept {}

  inline CodlOpType op_type() const {
    return op_type_;
  }

  inline int part_dim() const {
    return part_dim_;
  }

  inline void set_part_dim(int d) {
    part_dim_ = d;
  }

  inline float part_ratio() const {
    return part_ratio_;
  }

  inline void set_part_ratio(float r) {
    part_ratio_ = r;
  }

  inline int num_threads() const {
    return common_param_.num_threads;
  }

  inline DataType cpu_dtype() const {
    return common_param_.cpu_dtype;
  }

  inline DataType gpu_dtype() const {
    return common_param_.gpu_dtype;
  }

  inline MemoryType gpu_mtype() const {
    return common_param_.gpu_mtype;
  }

  inline CodlOpChainCommonParam common_param() const {
    return common_param_;
  }

  virtual CodlOpChainParam *Copy() = 0;

  virtual void CopyFrom(const CodlOpChainParam *param) {
    LOG(INFO) << "CodlOpChainParam: CopyFrom";
    if (param == nullptr) {
      return;
    }

    op_type_ = param->op_type();
    part_dim_ = param->part_dim();
    part_ratio_ = param->part_ratio();
    common_param_ = param->common_param();
  }
 
 protected:
  CodlOpType op_type_;
  int part_dim_;
  float part_ratio_;
  CodlOpChainCommonParam common_param_;
};

class CodlConv2dChainParam : public CodlOpChainParam {
 public:
  CodlConv2dChainParam(const index_t height,
                       const index_t width,
                       const index_t in_channel,
                       const index_t out_channel,
                       const index_t filter_height,
                       const index_t filter_width,
                       const int stride_h,
                       const int stride_w,
                       const int part_dim,
                       const float part_ratio,
                       const CodlOpChainCommonParam &common_param)
      : CodlOpChainParam(part_dim, part_ratio, common_param),
        height_(height),
        width_(width),
        in_channel_(in_channel),
        out_channel_(out_channel),
        filter_height_(filter_height),
        filter_width_(filter_width),
        stride_h_(stride_h),
        stride_w_(stride_w) {
    op_type_ = CODL_OP_TYPE_CONV2D;
  }

  inline index_t height() const {
    return height_;
  }

  inline void set_height(index_t height) {
    height_ = height;
  }

  inline index_t width() const {
    return width_;
  }

  inline void set_width(index_t width) {
    width_ = width;
  }

  inline index_t in_channel() const {
    return in_channel_;
  }

  inline void set_in_channel(index_t in_channel) {
    in_channel_ = in_channel;
  }

  inline index_t out_channel() const {
    return out_channel_;
  }

  inline void set_out_channel(index_t out_channel) {
    out_channel_ = out_channel;
  }

  inline index_t filter_height() const {
    return filter_height_;
  }

  inline void set_filter_height(index_t filter_height) {
    filter_height_ = filter_height;
  }

  inline index_t filter_width() const {
    return filter_width_;
  }

  inline void set_filter_width(index_t filter_width) {
    filter_width_ = filter_width;
  }

  inline int stride_h() const {
    return stride_h_;
  }

  inline void set_stride_h(int stride_h) {
    stride_h_ = stride_h;
  }

  inline int stride_w() const {
    return stride_w_;
  }

  inline void set_stride_w(int stride_w) {
    stride_w_ = stride_w;
  }

  CodlOpChainParam *Copy() override {
    return new CodlConv2dChainParam(height_, width_, in_channel_, out_channel_,
                                    filter_height_, filter_width_,
                                    stride_h_, stride_w_, part_dim_, part_ratio_,
                                    common_param_);
  }

  void CopyFrom(const CodlOpChainParam *param) override {
    LOG(INFO) << "CodlConv2dChainParam: CopyFrom";
    if (param == nullptr) {
      return;
    }
    CodlOpChainParam::CopyFrom(param);

    const CodlConv2dChainParam *conv2d_param =
        reinterpret_cast<const CodlConv2dChainParam *>(param);
    height_ = conv2d_param->height();
    width_ = conv2d_param->width();
    in_channel_ = conv2d_param->in_channel();
    out_channel_ = conv2d_param->out_channel();
    filter_height_ = conv2d_param->filter_height();
    filter_width_ = conv2d_param->filter_width();
    stride_h_ = conv2d_param->stride_h();
    stride_w_ = conv2d_param->stride_w();
  }
  
 private:
  index_t height_;
  index_t width_;
  index_t in_channel_;
  index_t out_channel_;
  index_t filter_height_;
  index_t filter_width_;
  int stride_h_;
  int stride_w_;
};

class CodlPoolingChainParam : public CodlOpChainParam {
 public:
  CodlPoolingChainParam(const index_t height,
                        const index_t width,
                        const index_t in_channel,
                        const index_t filter_height,
                        const index_t filter_width,
                        const int stride_h,
                        const int stride_w,
                        const int pooling_type,
                        const int part_dim,
                        const float part_ratio,
                        const CodlOpChainCommonParam &common_param)
      : CodlOpChainParam(part_dim, part_ratio, common_param),
        height_(height),
        width_(width),
        in_channel_(in_channel),
        filter_height_(filter_height),
        filter_width_(filter_width),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pooling_type_(pooling_type) {
    op_type_ = CODL_OP_TYPE_POOLING;
  }

  inline index_t height() const {
    return height_;
  }

  inline void set_height(index_t height) {
    height_ = height;
  }

  inline index_t width() const {
    return width_;
  }

  inline void set_width(index_t width) {
    width_ = width;
  }

  inline index_t in_channel() const {
    return in_channel_;
  }

  inline void set_in_channel(index_t in_channel) {
    in_channel_ = in_channel;
  }

  inline index_t filter_height() const {
    return filter_height_;
  }

  inline void set_filter_height(index_t filter_height) {
    filter_height_ = filter_height;
  }

  inline index_t filter_width() const {
    return filter_width_;
  }

  inline void set_filter_width(index_t filter_width) {
    filter_width_ = filter_width;
  }

  inline int stride_h() const {
    return stride_h_;
  }

  inline void set_stride_h(int stride_h) {
    stride_h_ = stride_h;
  }

  inline int stride_w() const {
    return stride_w_;
  }

  inline void set_stride_w(int stride_w) {
    stride_w_ = stride_w;
  }

  inline int pooling_type() const {
    return pooling_type_;
  }

  inline void set_pooling_type(const int pooling_type) {
    pooling_type_ = pooling_type;
  }

  CodlOpChainParam *Copy() override {
    return new CodlPoolingChainParam(
        height_, width_, in_channel_, filter_height_, filter_width_,
        stride_h_, stride_w_, pooling_type_, part_dim_, part_ratio_,
        common_param_);
  }

  void CopyFrom(const CodlOpChainParam *param) override {
    LOG(INFO) << "CodlConv2dChainParam: CopyFrom";
    if (param == nullptr) {
      return;
    }
    CodlOpChainParam::CopyFrom(param);

    const CodlPoolingChainParam *pooling_param =
        reinterpret_cast<const CodlPoolingChainParam *>(param);
    height_ = pooling_param->height();
    width_ = pooling_param->width();
    in_channel_ = pooling_param->in_channel();
    filter_height_ = pooling_param->filter_height();
    filter_width_ = pooling_param->filter_width();
    stride_h_ = pooling_param->stride_h();
    stride_w_ = pooling_param->stride_w();
    pooling_type_ = pooling_param->pooling_type();
  }
  
 private:
  index_t height_;
  index_t width_;
  index_t in_channel_;
  index_t filter_height_;
  index_t filter_width_;
  int stride_h_;
  int stride_w_;
  int pooling_type_;
};

class CodlFullyConnectedChainParam : public CodlOpChainParam {
 public:
  CodlFullyConnectedChainParam(const index_t height,
                               const index_t width,
                               const index_t in_channel,
                               const index_t out_channel,
                               const int part_dim,
                               const float part_ratio,
                               const CodlOpChainCommonParam &common_param)
      : CodlOpChainParam(part_dim, part_ratio, common_param),
        height_(height),
        width_(width),
        in_channel_(in_channel),
        out_channel_(out_channel) {
    op_type_ = CODL_OP_TYPE_FULLY_CONNECTED;
  }

  inline index_t height() const {
    return height_;
  }

  inline void set_height(index_t height) {
    height_ = height;
  }

  inline index_t width() const {
    return width_;
  }

  inline void set_width(index_t width) {
    width_ = width;
  }

  inline index_t in_channel() const {
    return in_channel_;
  }

  inline void set_in_channel(index_t in_channel) {
    in_channel_ = in_channel;
  }

  inline index_t out_channel() const {
    return out_channel_;
  }

  inline void set_out_channel(index_t out_channel) {
    out_channel_ = out_channel;
  }

  CodlOpChainParam *Copy() override {
    return new CodlFullyConnectedChainParam(
        height_, width_, in_channel_, out_channel_,
        part_dim_, part_ratio_, common_param_);
  }

  void CopyFrom(const CodlOpChainParam *param) override {
    LOG(INFO) << "CodlConv2dChainParam: CopyFrom";
    if (param == nullptr) {
      return;
    }
    CodlOpChainParam::CopyFrom(param);

    const CodlFullyConnectedChainParam *fc_param =
        reinterpret_cast<const CodlFullyConnectedChainParam *>(param);
    height_ = fc_param->height();
    width_ = fc_param->width();
    in_channel_ = fc_param->in_channel();
    out_channel_ = fc_param->out_channel();
  }
  
 private:
  index_t height_;
  index_t width_;
  index_t in_channel_;
  index_t out_channel_;
};

class CodlMatMulChainParam : public CodlOpChainParam {
 public:
  CodlMatMulChainParam(const index_t batch,
                       const index_t height,
                       const index_t width,
                       const index_t depth,
                       const bool transpose_a,
                       const bool transpose_b,
                       const int part_dim,
                       const float part_ratio,
                       const CodlOpChainCommonParam &common_param)
      : CodlOpChainParam(part_dim, part_ratio, common_param),
        batch_(batch), height_(height), width_(width), depth_(depth),
        transpose_a_(transpose_a), transpose_b_(transpose_b) {
    if (batch == 1) {
      part_dim_ = 4;
    } else if (batch > 1) {
      part_dim_ = 1;
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    op_type_ = CODL_OP_TYPE_MATMUL;
  };

  inline index_t batch() const {
    return batch_;
  }

  inline index_t height() const {
    return height_;
  }

  inline index_t width() const {
    return width_;
  }

  inline index_t depth() const {
    return depth_;
  }

  inline bool transpose_a() const {
    return transpose_a_;
  }

  inline bool transpose_b() const {
    return transpose_b_;
  }

  CodlOpChainParam *Copy() override {
    return new CodlMatMulChainParam(batch_, height_, width_, depth_,
                                    transpose_a_, transpose_b_,
                                    part_dim_, part_ratio_,
                                    common_param_);
  }

  void CopyFrom(const CodlOpChainParam *param) override {
    if (param == nullptr) {
      return;
    }
    CodlOpChainParam::CopyFrom(param);

    const CodlMatMulChainParam *matmul_param =
        reinterpret_cast<const CodlMatMulChainParam *>(param);
    batch_ = matmul_param->batch();
    height_ = matmul_param->height();
    width_ = matmul_param->width();
    depth_ = matmul_param->depth();
    transpose_a_ = matmul_param->transpose_a();
    transpose_b_ = matmul_param->transpose_b();
  }

 private:
  index_t batch_;
  index_t height_;
  index_t width_;
  index_t depth_;
  bool transpose_a_;
  bool transpose_b_;
};

class CodlOpTaskChain {
 public:
  CodlOpTaskChain() : is_ready_(false) {}

  int AddCommonTestParam(const int part_dim,
                         const float part_ratio,
                         const CodlOpChainCommonParam &common_param,
                         TestParam *param);

  int AppendConv2d(const index_t height,
                   const index_t width,
                   const index_t in_channel,
                   const index_t out_channel,
                   const index_t filter_height,
                   const index_t filter_width,
                   const int stride_h,
                   const int stride_w,
                   const int part_dim,
                   const float part_ratio,
                   const CodlOpChainCommonParam &common_param);

  int AppendPooling(const index_t height,
                    const index_t width,
                    const index_t in_channel,
                    const index_t filter_height,
                    const index_t filter_width,
                    const int stride_h,
                    const int stride_w,
                    const int pooling_type,
                    const int part_dim,
                    const float part_ratio,
                    const CodlOpChainCommonParam &common_param);

  int AppendFullyConnected(const index_t height,
                           const index_t width,
                           const index_t in_channel,
                           const index_t out_channel,
                           const int part_dim,
                           const float part_ratio,
                           const CodlOpChainCommonParam &common_param);

  int AppendMatMul(const index_t batch,
                   const index_t height,
                   const index_t width,
                   const index_t depth,
                   const bool transpose_a,
                   const bool transpose_b,
                   const int part_dim,
                   const float part_ratio,
                   const CodlOpChainCommonParam &common_param);

  int Append(const CodlOpChainParam *param) {
    //LOG(INFO) << "Append op type " << CodlOpTypeToString(param->op_type());
    if (param->op_type() == CODL_OP_TYPE_CONV2D) {
      const CodlConv2dChainParam *param_ptr =
          reinterpret_cast<const CodlConv2dChainParam *>(param);
      return AppendConv2d(param_ptr->height(), param_ptr->width(),
                          param_ptr->in_channel(), param_ptr->out_channel(),
                          param_ptr->filter_height(), param_ptr->filter_width(),
                          param_ptr->stride_h(), param_ptr->stride_w(),
                          param_ptr->part_dim(), param_ptr->part_ratio(),
                          param_ptr->common_param());
    } else if (param->op_type() == CODL_OP_TYPE_POOLING) {
      const CodlPoolingChainParam *param_ptr =
          reinterpret_cast<const CodlPoolingChainParam *>(param);
      return AppendPooling(param_ptr->height(), param_ptr->width(),
                           param_ptr->in_channel(),
                           param_ptr->filter_height(), param_ptr->filter_width(),
                           param_ptr->stride_h(), param_ptr->stride_w(),
                           param_ptr->pooling_type(),
                           param_ptr->part_dim(), param_ptr->part_ratio(),
                           param_ptr->common_param());
    } else if (param->op_type() == CODL_OP_TYPE_FULLY_CONNECTED) {
      const CodlFullyConnectedChainParam *param_ptr =
          reinterpret_cast<const CodlFullyConnectedChainParam *>(param);
      return AppendFullyConnected(param_ptr->height(), param_ptr->width(),
                                  param_ptr->in_channel(), param_ptr->out_channel(),
                                  param_ptr->part_dim(), param_ptr->part_ratio(),
                                  param_ptr->common_param());
    } else if (param->op_type() == CODL_OP_TYPE_MATMUL) {
      const CodlMatMulChainParam *param_ptr =
          reinterpret_cast<const CodlMatMulChainParam *>(param);
      return AppendMatMul(param_ptr->batch(), param_ptr->height(),
                          param_ptr->width(), param_ptr->depth(),
                          param_ptr->transpose_a(), param_ptr->transpose_b(),
                          param_ptr->part_dim(), param_ptr->part_ratio(),
                          param_ptr->common_param());
    } else {
      LOG(ERROR) << "Unsupported op type: " << static_cast<int>(param->op_type());
      MACE_NOT_IMPLEMENTED;
      return 0;
    }
  }

  inline bool is_ready() const {
    return is_ready_;
  }

  inline size_t size() const {
    return tasks_.size();
  }

  inline int dim() const {
    if (tasks_.size() > 0) {
      return static_cast<int>(tasks_[0]->part_plan()->dimension());
    }
    
    return 0;
  }

  inline float ratio() const {
    if (tasks_.size() > 0) {
      return tasks_[0]->part_plan()->ratio();
    }
    
    return 1;
  }

  index_t max_cpu_input_raw_size() const {
    index_t max_raw_size = -1;
    for (auto &task : tasks_) {
      if (task->cpu_input_raw_size() > max_raw_size) {
        max_raw_size = task->cpu_input_raw_size();
      }
    }
    MACE_CHECK(max_raw_size > -1);
    return max_raw_size;
  }

  index_t cpu_weight_raw_size() const {
    index_t raw_size = 0;
    for (auto &task : tasks_) {
      raw_size += task->cpu_weight_raw_size();
    }
    return raw_size;
  }

  index_t max_cpu_output_raw_size() const {
    index_t max_raw_size = -1;
    for (auto &task : tasks_) {
      if (task->cpu_output_raw_size() > max_raw_size) {
        max_raw_size = task->cpu_output_raw_size();
      }
    }
    MACE_CHECK(max_raw_size > -1);
    return max_raw_size;
  }

  index_t gpu_input_raw_size() const {
    index_t raw_size = 0;
    for (auto &task : tasks_) {
      raw_size += task->gpu_input_raw_size();
    }
    return raw_size;
  }

  index_t gpu_output_raw_size() const {
    index_t raw_size = 0;
    for (auto &task : tasks_) {
      raw_size += task->gpu_output_raw_size();
    }
    return raw_size;
  }

  index_t weight_raw_size() const {
    index_t raw_size = 0;
    for (auto &task : tasks_) {
      raw_size += task->weight_raw_size();
    }
    return raw_size;
  }

  std::shared_ptr<CodlOpCpuGpuTestTask> GetTask(const size_t i) const {
    if (i < tasks_.size()) {
      return tasks_[i];
    } else {
      return nullptr;
    }
  }

  int RemoveLast() {
    if (tasks_.size() > 0) {
      tasks_.pop_back();
    }
    return 0;
  }

  inline int Clear() {
    tasks_.clear();
    return 0;
  }

  inline int CopyFrom(const CodlOpTaskChain *chain) {
    Clear();
    for (size_t i = 0; i < chain->size(); i ++) {
      tasks_.push_back(chain->GetTask(i));
    }
    return 0;
  }

  inline bool Equal(const CodlOpTaskChain *chain) const {
    if (size() != chain->size()) {
      return false;
    }
    if (dim() != chain->dim()) {
      return false;
    }
    if (ratio() != chain->ratio()) {
      return false;
    }
    return true;
  }

  int Prepare();

  int SerialRun(DurationCollector<double> *collector) const;

  int SerialRun(const int si,
                const int rounds,
                const bool is_compute_only,
                double *lat) const;

  int SerialRun(double *lat) const;

  int SerialRun() const {
    double lat;
    return SerialRun(&lat);
  }

  int Run(DurationCollector<double> *collector) const;

  int Run(const int si,
          const int rounds,
          const bool is_compute_only,
          double *lat) const;

  int Run(double *lat) const;

  int Run() const {
    double lat;
    return Run(&lat);
  }

  int Destroy(const CodlTestTaskDestroyType type) {
    is_ready_ = false;
    for (auto &task : tasks_) {
      task->Destroy(type);
    }
    return 0;
  }

 protected:
  int UpdatePartitionShape();

  bool is_ready_;
  std::vector<std::shared_ptr<CodlOpCpuGpuTestTask>> tasks_;
};

}  // namespace mace

#endif  // TEST_CODL_RUN_OP_TEST_TASK_CHAIN_H_
