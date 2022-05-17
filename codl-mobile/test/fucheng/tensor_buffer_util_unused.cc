
Tensor* TensorUtils::CreateBufferTensor(xxx) {
  if (device_type == DeviceType::CPU) {
    //t = new Tensor(GetCPUAllocator(), dt, is_weight, name);
  }
}

Tensor* TensorUtils::CreatePartTensorV2(xxx) {

  //fprintf(stderr, "Info: Create part tensor offset=%ld size=%ld\n",
  //                    reuse_data_offset, reuse_data_size);
}

void FillTensorDataFloatFormatNHWC(xxx) {
  // Channels must be same
  //MACE_CHECK(static_cast<index_t>(template_data.size()) == tensor->dim(3));
  // Create temporary buffer
  //fprintf(stderr, "Info: Tensor raw size %ld\n", tensor->raw_size());
  //float *tensor_temp_data = new float[tensor->size()];
  //float *tensor_temp_data = static_cast<float*>(
  //    malloc(tensor->raw_size() + EXTRA_BUFFER_PAD_SIZE));

  for (int n = 0; n < tensor->dim(0); n ++) {
    for (int h = 0; h < tensor->dim(1); h ++) {
      for (int w = 0; w < tensor->dim(2); w ++) {
        for (int c = 0; c < tensor->dim(3); c ++) {
        }
      }

      //next_template_data.clear();
      //for (size_t i = 0; i < template_data.size(); i ++) {
      //    next_template_data.push_back(
      //        template_data[i] + (h + 1) * template_data.size());
      //}
    }
  }

  //PrintTempDataFloat(tensor_temp_data, tensor->size());

  // Delete temporary buffer.
  //delete[] tensor_temp_data;
  //free(tensor_temp_data);
}

void FillTensorDataFloatFormatNCHW(xxx) {
  // Channels must be same
  //MACE_CHECK(static_cast<index_t>(template_data.size()) == tensor->dim(1));
  // Create temporary buffer
  //fprintf(stderr, "Info: Tensor raw size %ld\n", tensor->raw_size());
  //float *tensor_temp_data = new float[tensor->size()];
  //float *tensor_temp_data = static_cast<float*>(
  //    malloc(tensor->raw_size() + EXTRA_BUFFER_PAD_SIZE));

  for (int n = 0; n < tensor->dim(0); n ++) {
    for (int c = 0; c < tensor->dim(1); c ++) {
      for (int h = 0; h < tensor->dim(2); h ++) {
        for (int w = 0; w < tensor->dim(3); w ++) {

          //float data = template_data[c];
          //data = data * 1.0f;
          //printf("Info: [%d,%d,%d,%d]\n", n, c, h, w);
        }
      }
    }

    /**
    // Update data?
    next_template_data.clear();
    for (size_t i = 0; i < template_data.size(); i ++) {
        next_template_data.push_back(
            template_data[i] + (n + 1) * template_data.size());
    }*/
  }

  //PrintTempDataFloat(tensor_temp_data,  tensor->size());

  // Delete temporary buffer.
  //delete[] tensor_temp_data;
  //free(tensor_temp_data);
}

void FillTensorDataFloatFormatNONE(xxx) {
  //float *tensor_temp_data = static_cast<float*>(
  //    malloc(tensor->raw_size() + EXTRA_BUFFER_PAD_SIZE));

  //free(tensor_temp_data);
}

void FillTensorDataFloat(xxx) {
  //PrintDataTemplateFloat(template_data);

  if (tensor->data_format() == DataFormat::NHWC) {
    //fprintf(stderr, "Info: Fill tensor data NHWC\n");
  } else if (tensor->data_format() == DataFormat::NCHW ||
             tensor->data_format() == DataFormat::OIHW) {
    //fprintf(stderr, "Info: Fill tensor data NCHW\n");
  } else {
    //fprintf(stderr, "Error: Not support tensor format\n");
  }
}
