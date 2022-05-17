
#include "mace/ops/opencl/buffer_transformer.h"
#include "test/fucheng/tensor_transpose_util.h"

namespace mace {

std::vector<int64_t> TensorTransposeUtil::ShapeTranspose(
    const std::vector<int64_t> &input_shape,
    const std::vector<int> &dst_dims) {
  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape.at(dst_dims.at(i)));
  }
  //LOG(INFO) << "Transpose shape from "
  //          << VectorToString<int64_t>(input_shape)
  //          << " to " << VectorToString<int64_t>(output_shape);
  return output_shape;
}

template<typename T1, typename T2>
int TensorTransposeUtil::TransposeImage(
    utils::ThreadPool *thread_pool,
    const T1 *input,
    const std::vector<int64_t> &input_shape,
    const std::vector<int> &dst_dims,
    OdimRanges &odim_ranges,
    T2 *output) {
  MACE_UNUSED(thread_pool);
  
  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape[dst_dims[i]]);
  }
  
  MACE_CHECK(odim_ranges.size() == 4);
  for (size_t i = 0; i < odim_ranges.size(); ++i) {
    if (odim_ranges[i].empty()) {
      odim_ranges[i].push_back(0);
      odim_ranges[i].push_back(output_shape[i]);
      odim_ranges[i].push_back(0);
    }
    
    MACE_CHECK(odim_ranges[i].size() == 3);
    
    //LOG(INFO) << "Odim range[" << i << "]: "
    //          << VectorToString(odim_ranges[i]);
  }
  
  // Update output shape.
  output_shape.clear();
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(odim_ranges[i][1] - odim_ranges[i][0]);
  }
  
  //LOG(INFO) << "Shape " << VectorToString(input_shape)
  //          << " to " << VectorToString(output_shape);
  
  std::function<T2(T1)> cast_func;
  if (typeid(T2) == typeid(half)) {
    cast_func = [&](T1 data) -> T2 {
      return half_float::half_cast<half>(data);
    };
  } else {
    cast_func = [&](T1 data) -> T2 {
      return static_cast<T2>(data);
    };
  }
  
  //LOG(INFO) << "Typename " << typeid(T1).name()
  //          << " to " << typeid(T2).name();
  
  if (input_shape.size() == 4) {
    std::vector<index_t>
        in_stride{input_shape[1] * input_shape[2] * input_shape[3],
                  input_shape[2] * input_shape[3],
                  input_shape[3],
                  1};
    std::vector<index_t>
        out_stride{output_shape[1] * output_shape[2] * output_shape[3],
                   output_shape[2] * output_shape[3],
                   output_shape[3],
                   1};

    std::vector<index_t> idim(4, 0);
    std::vector<index_t> odim(4, 0);
    for (odim[0] = odim_ranges[0][0];
        odim[0] < odim_ranges[0][1]; ++odim[0]) {
      for (odim[1] = odim_ranges[1][0];
            odim[1] < odim_ranges[1][1]; ++odim[1]) {
        for (odim[2] = odim_ranges[2][0];
            odim[2] < odim_ranges[2][1]; ++odim[2]) {
          for (odim[3] = odim_ranges[3][0];
                odim[3] < odim_ranges[3][1]; ++odim[3]) {
            idim[dst_dims[0]] = odim[0] + odim_ranges[0][2];
            idim[dst_dims[1]] = odim[1] + odim_ranges[1][2];
            idim[dst_dims[2]] = odim[2] + odim_ranges[2][2];
            idim[dst_dims[3]] = odim[3] + odim_ranges[3][2];
            
            //LOG(INFO) << VectorToString(idim)
            //          << " to " << VectorToString(odim);
            
            //T1 input_v  = input[idim[0] * in_stride[0] +
            //                    idim[1] * in_stride[1] +
            //                    idim[2] * in_stride[2] +
            //                    idim[3]];
            
            // cast_func() or static_cast<T2>()
            output[odim[0] * out_stride[0] +
                   odim[1] * out_stride[1] +
                   odim[2] * out_stride[2] +
                   odim[3]]
                   = cast_func(input[idim[0] * in_stride[0] +
                                     idim[1] * in_stride[1] +
                                     idim[2] * in_stride[2] +
                                     idim[3]]);

            //output[odim[0] * out_stride[0] + odim[1] * out_stride[1]
            //         + odim[2] * out_stride[2] + odim[3]] =
            //    input[idim[0] * in_stride[0] + idim[1] * in_stride[1]
            //            + idim[2] * in_stride[2] + idim[3]];
          }
        }
      }
    }
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  
  return 0;
}

template<typename T1, typename T2>
int TensorTransposeUtil::TransposeImageToBufferConv2dFilter(
    utils::ThreadPool *thread_pool,
    const T1 *input,
    const std::vector<int64_t> &input_shape,  // IOHW
    const std::vector<int> &dst_dims,
    OdimRanges &odim_ranges,
    T2 *output) {
  MACE_UNUSED(thread_pool);
  
  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape[dst_dims[i]]);
  }
  
  MACE_CHECK(odim_ranges.size() == 4);
  for (size_t i = 0; i < odim_ranges.size(); ++i) {
    if (odim_ranges[i].empty()) {
      odim_ranges[i].push_back(0);
      odim_ranges[i].push_back(output_shape[i]);
      odim_ranges[i].push_back(0);
    }
    
    MACE_CHECK(odim_ranges[i].size() == 3);
    
    LOG(INFO) << "Odim range[" << i << "]: " << VectorToString(odim_ranges[i]);
  }
  
  // Update output shape.
  output_shape.clear();
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(odim_ranges[i][1] - odim_ranges[i][0]);
  }
  
  LOG(INFO) << "Shape " << VectorToString(input_shape)
            << " to " << VectorToString(output_shape);
  
  std::function<T2(T1)> cast_func;
  if (typeid(T2) == typeid(half)) {
    cast_func = [&](T1 data) -> T2 {
      return half_float::half_cast<half>(data);
    };
  } else {
    cast_func = [&](T1 data) -> T2 {
      return static_cast<T2>(data);
    };
  }
  
  LOG(INFO) << "Typename " << typeid(T1).name() << " to " << typeid(T2).name();
  
  if (input_shape.size() == 4) {
    std::vector<index_t>
        in_stride{input_shape[1] * input_shape[2] * input_shape[3],
                  input_shape[2] * input_shape[3],
                  input_shape[3],
                  1};
    std::vector<index_t>
        out_stride{output_shape[1] * output_shape[2] * output_shape[3],
                   output_shape[2] * output_shape[3],
                   output_shape[3],
                   1};
                   
    std::vector<index_t> input_offset_stride;
    input_offset_stride.push_back(input_shape[0] * 4 * in_stride[1]);
    input_offset_stride.push_back(4);
    input_offset_stride.push_back(input_shape[0] * 4);

    std::vector<index_t> idim(4, 0);
    std::vector<index_t> odim(4, 0);
    index_t o4_count = 0;
    std::vector<index_t> input_offset(4, 0);
    for (odim[0] = odim_ranges[0][0]; odim[0] < odim_ranges[0][1]; ++odim[0]) {
      if ((odim[0] >> 2) > o4_count) {
        o4_count ++;
        input_offset[0] += input_offset_stride[0];
      }
      input_offset[1] = odim[0] % 4;
      for (odim[1] = odim_ranges[1][0];
           odim[1] < odim_ranges[1][1]; ++odim[1]) {
        input_offset[2] = 0;
        for (odim[2] = odim_ranges[2][0];
             odim[2] < odim_ranges[2][1]; ++odim[2]) {
          for (odim[3] = odim_ranges[3][0];
               odim[3] < odim_ranges[3][1]; ++odim[3]) {
            idim[dst_dims[0]] = odim[0] + odim_ranges[0][2];
            idim[dst_dims[1]] = odim[1] + odim_ranges[1][2];
            idim[dst_dims[2]] = odim[2] + odim_ranges[2][2];
            idim[dst_dims[3]] = odim[3] + odim_ranges[3][2];
            
            idim[1] -= (o4_count << 2);
            
            //LOG(INFO) << VectorToString(idim)
            //          << " to " << VectorToString(odim);
            
            /**
            const T1 *input_ptr  = input + input_offset[0]
                                         + input_offset[1]
                                         + input_offset[2];
            
            T2 *output_ptr = output + (odim[0] * out_stride[0] +
                                       odim[1] * out_stride[1] +
                                       odim[2] * out_stride[2] +
                                       odim[3]);
                                       
            *output_ptr = cast_func(*input_ptr);
            */
            
            output[odim[0] * out_stride[0] +
                   odim[1] * out_stride[1] +
                   odim[2] * out_stride[2] +
                   odim[3]] = cast_func(input[input_offset[0] +
                                              input_offset[1] +
                                              input_offset[2]]);
            
            input_offset[2] += input_offset_stride[2];
          }
        }
        
        input_offset[1] += input_offset_stride[1];
      }
    }
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  
  return 0;
}

template<typename T1, typename T2>
int TensorTransposeUtil::TransposeBufferToImageConv2dFilter(
    utils::ThreadPool *thread_pool,
    const T1 *input,
    const std::vector<int64_t> &input_shape,  // OIHW
    const std::vector<int> &dst_dims,
    OdimRanges &odim_ranges,
    T2 *output) {
  MACE_UNUSED(thread_pool);
  MACE_UNUSED(output);
  
  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape[dst_dims[i]]);
  }
  
  MACE_CHECK(odim_ranges.size() == 4);
  for (size_t i = 0; i < odim_ranges.size(); ++i) {
    if (odim_ranges[i].empty()) {
      odim_ranges[i].push_back(0);
      odim_ranges[i].push_back(output_shape[i]);
      odim_ranges[i].push_back(0);
    }
    
    MACE_CHECK(odim_ranges[i].size() == 3);
    
    LOG(INFO) << "Odim range[" << i << "]: " << VectorToString(odim_ranges[i]);
  }
  
  // Update output shape.
  output_shape.clear();
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(odim_ranges[i][1] - odim_ranges[i][0]);
  }
  
  LOG(INFO) << "Shape " << VectorToString(input_shape)
            << " to " << VectorToString(output_shape);
  
  std::function<T2(T1)> cast_func;
  if (typeid(T2) == typeid(half)) {
    cast_func = [&](T1 data) -> T2 {
      return half_float::half_cast<half>(data);
    };
  } else {
    cast_func = [&](T1 data) -> T2 {
      return static_cast<T2>(data);
    };
  }
  
  LOG(INFO) << "Typename " << typeid(T1).name() << " to " << typeid(T2).name();
  
  if (input_shape.size() == 4) {
    std::vector<index_t>
        in_stride{input_shape[1] * input_shape[2] * input_shape[3],
                  input_shape[2] * input_shape[3],
                  input_shape[3],
                  1};
    std::vector<index_t> out_stride{
        output_shape[2] * output_shape[3] * output_shape[0] * 4,
        output_shape[0] * 4,
        4,
        1};
    std::vector<index_t> idim(4, 0);
    std::vector<index_t> odim(5, 0);
    
    std::stringstream stream;
    stream << "[";
    
    for (odim[0] = 0; odim[0] < (output_shape[1] >> 2); odim[0] ++) {
      for (odim[1] = 0; odim[1] < output_shape[2]; odim[1] ++) {
        idim[2] = odim[1];
        for (odim[2] = 0; odim[2] < output_shape[3]; odim[2] ++) {
          idim[3] = odim[2];
          for (odim[3] = 0; odim[3] < output_shape[0]; odim[3] ++) {
            idim[1] = odim[3] % input_shape[1];
            for (odim[4] = 0; odim[4] < 4; odim[4] ++) {
              const T1 *input_ptr = input + (idim[0] * in_stride[0] +
                                             idim[1] * in_stride[1] +
                                             idim[2] * in_stride[2] +
                                             idim[3]);
                                             
              T2 *output_ptr
                = output + odim[0] * out_stride[0]
                  + (odim[1] * output_shape[2] + odim[2]) * out_stride[1]
                  + odim[3] * out_stride[2]
                  + odim[4];
                                             
              //stream << *input_ptr << ",";
              MACE_UNUSED(input_ptr);
              stream << *output_ptr << ",";
              
              idim[1] += input_shape[1];
            }
          }
        }
      }
      
      idim[0] += 4;
    }
    
    stream << "]";
    //LOG(INFO) << stream.str();
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  
  return 0;
}

int TensorTransposeUtil::TransposeImageV2(
    OpContext *context,
    Tensor *input,
    const MemoryType in_mem_type,
    const MemoryType out_mem_type,
    const OpenCLBufferType buffer_type,
    const int wino_block_size,
    const std::vector<int> &dst_dims,
    OdimRanges &odim_ranges,
    Tensor *output) {
  ops::OpenCLBufferTransformer(in_mem_type, out_mem_type)
      .TransformAndPartTranspose(context, input, buffer_type,
                                 out_mem_type, wino_block_size,
                                 dst_dims, odim_ranges, output);
  return 0;
}

int TensorTransposeUtil::TransposeImageV3(
    OpContext *context,
    const Tensor *input,
    const MemoryType in_mem_type,
    const MemoryType out_mem_type,
    const OpenCLBufferType buffer_type,
    const int wino_block_size,
    const std::vector<int> &dst_dims,
    OdimRanges &odim_ranges,
    Tensor *output) {
  ops::OpenCLPartBufferTransformer(in_mem_type, out_mem_type)
      .PartTransformNHWC(
          context, input, buffer_type, out_mem_type,
          wino_block_size, dst_dims, odim_ranges, output);
  
  return 0;
}

int TensorTransposeUtil::TransposeImageV4(
    OpContext *context,
    const Tensor *input,
    const MemoryType in_mem_type,
    const MemoryType out_mem_type,
    const OpenCLBufferType buffer_type,
    const int wino_block_size,
    OdimRanges &odim_ranges,
    Tensor *output) {
  ops::OpenCLPartBufferTransformer(in_mem_type, out_mem_type)
      .PartTransformNCHW(
          context, input, buffer_type,out_mem_type,
          wino_block_size, odim_ranges, output);
  
  return 0;
}

int TensorTransposeUtil::Transpose(
    OpContext *op_context,
    Tensor *src,
    Tensor *dst,
    OdimRanges *odim_ranges_ptr,
    const int wino_block_size,
    const bool use_default_mapping) {
  utils::ThreadPool *thread_pool
      = &op_context->device()->cpu_runtime()->thread_pool();
  
  MACE_CHECK(thread_pool != nullptr);
    
  //int64_t t0, t1;
  //double time_mills_map, time_mills_trans;
  
  //LOG(INFO) << "Mapping guard ready in TensorTranspose";
  
  AllocatorMapType map_type_src = AllocatorMapType::AMT_READ_ONLY;
  AllocatorMapType map_type_dst = AllocatorMapType::AMT_WRITE_ONLY;
  BlockFlag block_flag = BlockFlag::BF_FALSE;
  if (use_default_mapping) {
    map_type_src = AllocatorMapType::AMT_READ_WRITE;
    map_type_dst = AllocatorMapType::AMT_READ_WRITE;
    block_flag   = BlockFlag::BF_TRUE;
  }
  
  //t0 = NowMicros();
  Tensor::MappingGuard guard_src(src, map_type_src, block_flag);
  //LOG(INFO) << "Mapping guard src tensor OK in TensorTranspose";
  Tensor::MappingGuard guard_dst(dst, map_type_dst, block_flag);
  //LOG(INFO) << "Mapping guard dst tensor OK in TensorTranspose";
  //t1 = NowMicros();
  //time_mills_map = (t1 - t0) / 1000.0;
  
  //LOG(INFO) << "Mapping guard OK in TensorTranspose";
  
  // Determine destinate dims.
  std::vector<int> dst_dims;
  if (src->has_opencl_image() &&
      dst->data_format() == DataFormat::NCHW) {
    dst_dims = src->is_weight() ? DST_DIMS_IMAGE_TO_OIHW
                                : DST_DIMS_IMAGE_TO_NCHW;
  } else if (src->data_format() == DataFormat::NCHW &&
             dst->has_opencl_image()) {
    dst_dims = src->is_weight() ? DST_DIMS_OIHW_TO_IMAGE
                                : DST_DIMS_NCHW_TO_IMAGE;
  } else if (src->data_format() == DataFormat::NHWC &&
             dst->data_format() == DataFormat::NCHW) {
    dst_dims = DST_DIMS_NHWC_TO_NCHW;
  } else if (src->data_format() == DataFormat::NCHW &&
             dst->data_format() == DataFormat::NHWC) {
    dst_dims = DST_DIMS_NCHW_TO_NHWC;
  } else {
    LOG(ERROR) << "Unsupported tensor transpose "
               << DataFormatToString(src->data_format()) << " to "
               << DataFormatToString(dst->data_format());
    return 1;
  }
  
  //LOG(INFO) << "Dst dims OK in TensorTranspose";
  //LOG(INFO) << "Dst dims " << VectorToString<int>(dst_dims);
  //LOG(INFO) << "Destinate dims: " << VectorToString<int>(dst_dims);
  
  OdimRanges odim_ranges(4);
  if (odim_ranges_ptr != nullptr) {
    odim_ranges = *odim_ranges_ptr;
  }
  
  Tensor *input  = src;
  Tensor *output = dst;
  //t0 = NowMicros();
  if (src->has_opencl_image() &&
      !src->is_weight()       &&
      dst->data_format() == DataFormat::NCHW) {
    //const half *input_data = input->data<half>();
    //float *output_data = output->mutable_data<float>();
    //TransposeImage<half, float>(thread_pool, input_data, input->shape(),
    //                            dst_dims, odim_ranges, output_data);
    
    // V2 or V3
    TransposeImageV3(op_context, input,
                     MemoryType::GPU_IMAGE, MemoryType::CPU_BUFFER,
                     OpenCLBufferType::IN_OUT_CHANNEL,
                     wino_block_size, dst_dims, odim_ranges, output);
    
    //TransposeImageV4(op_context, input,
    //                 MemoryType::GPU_IMAGE, MemoryType::CPU_BUFFER,
    //                 OpenCLBufferType::IN_OUT_CHANNEL,
    //                 wino_block_size, odim_ranges, output);
  } else if (src->data_format() == DataFormat::NCHW &&
             !src->is_weight()                      &&
             dst->has_opencl_image()) {
    //const float *input_data = input->data<float>();
    //half *output_data = output->mutable_data<half>();
    
    //TransposeImage<float, half>(thread_pool, input_data, input->shape(),
    //                            dst_dims, odim_ranges, output_data);
    
    //if (input->data_format() == DataFormat::NCHW) {
        // NOTE: We should transpose input shape to NHWC.
        //       Because shape used in Transform is NHWC
        //input->Reshape(
        //    ShapeTranspose(input->shape(), DST_DIMS_NCHW_TO_NHWC));
        //input->set_data_format(DataFormat::NHWC);
    //}
    
    // V2 or V3
    TransposeImageV3(op_context, input,
                     MemoryType::CPU_BUFFER, MemoryType::GPU_IMAGE,
                     OpenCLBufferType::IN_OUT_CHANNEL,
                     wino_block_size, dst_dims, odim_ranges, output);
                     
    //TransposeImageV4(op_context, input,
    //                 MemoryType::CPU_BUFFER, MemoryType::GPU_IMAGE,
    //                 OpenCLBufferType::IN_OUT_CHANNEL,
    //                 wino_block_size, odim_ranges, output);
    
    // NOTE: Not need to transpose shape
    return 0;
  } else if (src->is_weight()) {
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();
    
    std::vector<int64_t> input_shape = input->shape();
    if (src->has_opencl_image()) {
      input_shape = ShapeTranspose(input_shape, DST_DIMS_OIHW_TO_IMAGE);
      // NOTE: input_shape here must be IMAGE (IOHW).
      TransposeImageToBufferConv2dFilter<float, float>(
          thread_pool, input_data, input_shape,
          dst_dims, odim_ranges, output_data);
    } else if (dst->has_opencl_image()) {
      TransposeBufferToImageConv2dFilter<float, float>(
          thread_pool, input_data, input_shape,
          dst_dims, odim_ranges, output_data);
    }
  } else {
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();
    
    //MACE_CHECK(input_data != nullptr);
    //MACE_CHECK(output_data != nullptr);
    
    //LOG(INFO) << "Transpose ready in TensorTranspose";
    
    std::vector<int64_t> input_shape = input->shape();
    // Transpose shape for tensor which is weight and has opencl image.
    if (src->has_opencl_image() && src->is_weight()) {
      input_shape = ShapeTranspose(input_shape, DST_DIMS_OIHW_TO_IMAGE);
    }
    
    mace::ops::Transpose<float>(thread_pool, input_data, input_shape,
                                dst_dims, output_data);
    
    //LOG(INFO) << "Transpose OK in TensorTranspose";
  }
  //t1 = NowMicros();
  //time_mills_trans = (t1 - t0) / 1000.0;
  //LOG(INFO) << "MappingGuard " << time_mills_map
  //          << " ms Transpose " << time_mills_trans << " ms";
  
  std::vector<index_t> transposed_output_shape;
  if (odim_ranges_ptr == nullptr) {
    // Update output tensor shape if they are different.
    transposed_output_shape = input->shape();
    if (src->has_opencl_image()) {
      // NOTE: Default data format of GPU Image Tensor is NHWC or OIHW,
      //       so we need to transpose it to Image format
      //       i.e. CWHN or IOHW first.
      transposed_output_shape
          = src->is_weight() ? ShapeTranspose(transposed_output_shape,
                                              DST_DIMS_OIHW_TO_IMAGE) :
                               //ShapeTranspose(transposed_output_shape,
                               //               DST_DIMS_NHWC_TO_IMAGE);
                               transposed_output_shape;
    }
  } else {
    transposed_output_shape = output->shape();
  }
  
  transposed_output_shape = ShapeTranspose(transposed_output_shape, dst_dims);
  if (output->shape() != transposed_output_shape) {
    std::vector<index_t> src_shape = output->shape();
    output->Reshape(transposed_output_shape);
    LOG(INFO) << "Output tensor shape is set from "
              << VectorToString<index_t>(src_shape) << " to "
              << VectorToString<index_t>(transposed_output_shape);
  }
  
  return 0;
}

}  // namespace name
