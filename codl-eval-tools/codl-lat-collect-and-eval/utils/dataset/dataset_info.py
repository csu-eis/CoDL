
import os
from enum import Enum

class DatasetName(Enum):
  FLOPsLRConv2d = 0
  MuLayerLRConv2d = 1
  MuLayerLRPooling = 2
  MuLayerLRFullyConnected = 3
  MLConv2d = 4
  MLConv2dOp = 5
  MLConv2dOpImplType = 6
  MLPooling = 7
  MLFullyConnected = 8
  MLFullyConnectedOp = 9
  MLDeconv2d = 10
  MLMatMul = 11
  MLMatMulOp = 12

def StringToDatasetName(output_dataset_type):
  if output_dataset_type == 'FLOPsLRConv2d':
    return DatasetName.FLOPsLRConv2d
  elif output_dataset_type == 'MuLayerLRConv2d':
    return DatasetName.MuLayerLRConv2d
  elif output_dataset_type == 'MuLayerLRPooling':
    return DatasetName.MuLayerLRPooling
  elif output_dataset_type == 'MuLayerLRFullyConnected':
    return DatasetName.MuLayerLRFullyConnected
  elif output_dataset_type == 'MLConv2d':
    return DatasetName.MLConv2d
  elif output_dataset_type == 'MLConv2dOp':
    return DatasetName.MLConv2dOp
  elif output_dataset_type == 'MLConv2dOpImplType':
    return DatasetName.MLConv2dOpImplType
  elif output_dataset_type == 'MLPooling':
    return DatasetName.MLPooling
  elif output_dataset_type == 'MLFullyConnected':
    return DatasetName.MLFullyConnected
  elif output_dataset_type == 'MLFullyConnectedOp':
    return DatasetName.MLFullyConnectedOp
  elif output_dataset_type == 'MLDeconv2d':
    return DatasetName.MLDeconv2d
  elif output_dataset_type == 'MLMatMul':
    return DatasetName.MLMatMul
  else:
    raise ValueError('Unsupported output dataset type: ' + output_dataset_type)

class DatasetInfo(object):
  def model_names(self):
    return None

  def data_filenames(self, model_names):
    path = 'dataset'
    data_filenames = []
    for model_name in model_names:
      filepath = os.path.join(path, model_name + '.csv')
      data_filenames.append(filepath)
    return data_filenames

  def all_data_filenames(self):
    return self.data_filenames(self.model_names())

  def col_names(self):
    return None

  def label_names(self):
    return None

  def feature_names(self):
    return None

class HousingDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['housing']

  def col_names(self):
    model_names = self.model_names()
    return {model_names[0]:
            'CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV'}

  def label_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['MEDV']}

  def feature_names(self):
    model_names = self.model_names()
    return {model_names[0]:
            ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
             'DIS','RAD','TAX','PTRATIO','B','LSTAT']}

class FLOPsLRConv2dDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_flops_cpu', 't_flops_gpu']

  def col_names(self):
    model_names = self.model_names()
    return {model_names[0]: 'T,FLOPS',
            model_names[1]: 'T,FLOPS'}

  def label_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['T'],
            model_names[1]: ['T']}

  def feature_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['FLOPS'],
            model_names[1]: ['FLOPS']}

class MuLayerLRConv2dDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_mulayer_conv2d_cpu', 't_mulayer_conv2d_gpu']

  def col_names(self):
    model_names = self.model_names()
    return {model_names[0]: 'COST,SD_COST,PDIM,PRATIO,FLOPS,F0,F1',
            model_names[1]: 'COST,SD_COST,PDIM,PRATIO,FLOPS,F0,F1'}

  def label_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['COST'],
            model_names[1]: ['COST']}

  def feature_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['FLOPS', 'F0','F1'],
            model_names[1]: ['FLOPS', 'F0','F1']}

class MuLayerLRPoolingDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_mulayer_pooling_cpu', 't_mulayer_pooling_gpu']

  def col_names(self):
    model_names = self.model_names()
    return {model_names[0]: 'COST,SD_COST,PDIM,PRATIO,F0,F1',
            model_names[1]: 'COST,SD_COST,PDIM,PRATIO,F0,F1'}

  def label_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['COST'],
            model_names[1]: ['COST']}

  def feature_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['F0','F1'],
            model_names[1]: ['F0','F1']}

class MuLayerLRFullyConnectedDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_mulayer_fc_cpu', 't_mulayer_fc_gpu']

  def col_names(self):
    model_names = self.model_names()
    return {model_names[0]: 'COST,SD_COST,PDIM,PRATIO,FLOPS,F0,F1',
            model_names[1]: 'COST,SD_COST,PDIM,PRATIO,FLOPS,F0,F1'}

  def label_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['COST'],
            model_names[1]: ['COST']}

  def feature_names(self):
    model_names = self.model_names()
    return {model_names[0]: ['FLOPS', 'F0','F1'],
            model_names[1]: ['FLOPS', 'F0','F1']}

GPU_ATOM_WAVE = 'WAVE'
GPU_ATOM_WARP = 'WARP'
GPU_ATOM = GPU_ATOM_WARP

class MLConv2dDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_flops_cpu_direct',
            't_pack_lb_cpu_gemm',
            't_pack_rb_cpu_gemm',
            't_comp_cpu_gemm',
            't_pad_cpu_winograd',
            't_unpad_cpu_winograd',
            't_tr_in_cpu_winograd',
            't_tr_out_cpu_winograd',
            't_{}_gpu_direct'.format(GPU_ATOM.lower())]

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'T_FLOPS,IH,IW,IC,OC,KH,KW,SH,SW',
                 model_names[1]: 'T_PACK_LB,M,K',
                 model_names[2]: 'T_PACK_RB,K,N',
                 model_names[3]: 'T_COMP,MB,KB,NB',
                 model_names[4]: 'T_PAD,IH,IW,IC',
                 model_names[5]: 'T_UNPAD,OH,OW,OC',
                 model_names[6]: 'T_TR_IN,PIH,PIW,OT',
                 model_names[7]: 'T_TR_OUT,POH,POW,OT',
                 model_names[8]: 'T_{},OH,OWB,ICB,OCB,KS,S'.format(GPU_ATOM)}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_FLOPS'],
                   model_names[1]: ['T_PACK_LB'],
                   model_names[2]: ['T_PACK_RB'],
                   model_names[3]: ['T_COMP'],
                   model_names[4]: ['T_PAD'],
                   model_names[5]: ['T_UNPAD'],
                   model_names[6]: ['T_TR_IN'],
                   model_names[7]: ['T_TR_OUT'],
                   model_names[8]: ['T_{}'.format(GPU_ATOM)]}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: ['IH','IW','IC','OC','KH','KW','SH','SW'],
                     model_names[1]: ['M','K'],
                     model_names[2]: ['K','N'],
                     model_names[3]: ['MB','KB','NB'],
                     model_names[4]: ['IH','IW','IC'],
                     model_names[5]: ['OH','OW','OC'],
                     model_names[6]: ['PIH','PIW','OT'],
                     model_names[7]: ['POH','POW','OT'],
                     model_names[8]: ['OH','OWB','ICB','OCB','KS','S']}
    return feature_names

class MLConv2dDatasetInfoV2(DatasetInfo):
  def model_names(self):
    model_names = ['t_data_sharing',
                   't_conv2d_cpu_direct',
                   't_conv2d_cpu_gemm',
                   't_conv2d_cpu_winograd',
                   't_conv2d_cpu_winograd_gemm',
                   't_conv2d_gpu_direct']
    return model_names

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'T_DT_IN,T_MAP_IN,T_MAP_OUT,T_SYNC,T_UNMAP_IN,T_UNMAP_OUT,T_DT_OUT,T_SYNC_2,' \
                                    + 'IH,IW,IC,OH,OW,OC,DATA_IN,DATA_OUT',
                 model_names[1]: 'CNN,COMP,T_FLOPS,SD_PAD,SD_COMP,SD_UNPAD,PDIM,PRATIO,' \
                                    + 'IH,IW,IC,OC,KH,KW,SH,SW,MAX_TILE_THR,FLOPS_PER_TILE',
                 model_names[2]: 'CNN,PACK_LB,PACK_RB,COMP,T_PACK_LB,T_PACK_RB,T_COMP,' \
                                    + 'SD_PACK_LB,SD_PACK_RB,SD_COMP,PDIM,PRATIO,' \
                                    + 'M,K,N,MB,KB,NB,MAX_MB_THR,MAX_NB_THR',
                 model_names[3]: 'CNN,PAD,TR_IN,TR_OUT,UNPAD,T_PAD,T_TR_IN,T_TR_OUT,T_UNPAD,' \
                                    + 'SD_PAD,SD_TR_IN,SD_COMP,SD_TR_OUT,SD_UNPAD,PDIM,PRATIO,' \
                                    + 'IH,IW,IC,OH,OW,OC,PIH,PIW,POH,POW,OT,MAX_IC_THR,MAX_OC_THR,TOTAL_OT_COUNT',
                 model_names[4]: 'CNN,PACK_LB,PACK_RB,COMP,T_PACK_LB,T_PACK_RB,T_COMP,' \
                                    + 'SD_PAD,SD_TR_IN,SD_COMP,SD_TR_OUT,SD_UNPAD,PDIM,PRATIO,' \
                                    + 'M,K,N,MB,KB,NB,MAX_MB_THR,MAX_NB_THR,OT',
                 model_names[5]: ('CNN,COMP,T_{},T_{}_ICB,SD_COMP,PDIM,PRATIO,' \
                                    + 'OH,OWB,ICB,OCB,KS,S,N_{}').format(GPU_ATOM, GPU_ATOM, GPU_ATOM)}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_DT_IN', 'T_MAP_IN', 'T_MAP_OUT', 'T_SYNC', 'T_UNMAP_IN', 'T_UNMAP_OUT', 'T_DT_OUT'],
                   model_names[1]: ['T_FLOPS'],
                   model_names[2]: ['T_PACK_LB', 'T_PACK_RB', 'T_COMP'],
                   model_names[3]: ['T_PAD', 'T_UNPAD', 'T_TR_IN', 'T_TR_OUT'],
                   model_names[4]: ['T_PACK_LB', 'T_PACK_RB', 'T_COMP'],
                   model_names[5]: ['T_{}'.format(GPU_ATOM)]}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT'],
                                      ['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT'],
                                      ['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT'],
                                      ['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT'],
                                      ['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT'],
                                      ['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT'],
                                      ['IH', 'IW', 'IC', 'OH', 'OW', 'OC', 'DATA_IN', 'DATA_OUT']],
                     model_names[1]: [['IH','IW','IC','OC','KH','KW','SH','SW']],
                     model_names[2]: [['M','K'], ['K','N'], ['M','K','N']],
                     model_names[3]: [['IH','IW','IC'], ['PIH','PIW','OT'], ['POH','POW','OT'], ['OH','OW','OC']],
                     model_names[4]: [['M','K'], ['K','N'], ['M','K','N']],
                     model_names[5]: [['OH','OWB','ICB','OCB','KS','S']]}
    return feature_names

class MLConv2dOpDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_conv2d_op_cpu',
            't_conv2d_op_gpu']

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,T_OP,SD_T_OP,PDIM,PRATIO,' \
                                 + 'IH,IW,IC,OC,KH,KW,SH,SW,DH,DW,FLOPS,PARAMS',
                 model_names[1]: 'CNN,T_OP,SD_T_OP,PDIM,PRATIO,' \
                                 + 'IH,IW,IC,OC,KH,KW,SH,SW,DH,DW,FLOPS,PARAMS'}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_OP'],
                   model_names[1]: ['T_OP']}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW', 'DH', 'DW', 'FLOPS', 'PARAMS']],
                     model_names[1]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW', 'DH', 'DW', 'FLOPS', 'PARAMS']]}
    return feature_names

class MLConv2dOpImplTypeDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_op_cpu_direct',
            't_op_cpu_gemm',
            't_op_cpu_winograd',
            't_op_gpu_direct']

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,T_OP,IH,IW,IC,OC,KH,KW,SH,SW,DH,DW,FLOPS,PARAMS',
                 model_names[1]: 'CNN,T_OP,IH,IW,IC,OC,KH,KW,SH,SW,DH,DW,FLOPS,PARAMS',
                 model_names[2]: 'CNN,T_OP,IH,IW,IC,OC,KH,KW,SH,SW,DH,DW,FLOPS,PARAMS',
                 model_names[3]: 'CNN,T_OP,IH,IW,IC,OC,KH,KW,SH,SW,DH,DW,FLOPS,PARAMS'}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_OP'],
                   model_names[1]: ['T_OP'],
                   model_names[2]: ['T_OP'],
                   model_names[3]: ['T_OP']}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW', 'DH', 'DW', 'FLOPS', 'PARAMS']],
                     model_names[1]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW', 'DH', 'DW', 'FLOPS', 'PARAMS']],
                     model_names[2]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW', 'DH', 'DW', 'FLOPS', 'PARAMS']],
                     model_names[3]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW', 'DH', 'DW', 'FLOPS', 'PARAMS']]}
    return feature_names

class MLPoolingDatasetInfo(DatasetInfo):
  def model_names(self):
    model_names = ['t_pooling_cpu_direct_avg',
                   't_pooling_cpu_direct_max',
                   't_pooling_gpu_direct_avg',
                   't_pooling_gpu_direct_max']
    return model_names

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,COST,T_COST,SD_COST,PDIM,PRATIO,' \
                                    + 'IH,IW,IC,OC,KH,KW,SH,SW',
                 model_names[1]: 'CNN,COST,T_COST,SD_COST,PDIM,PRATIO,' \
                                    + 'IH,IW,IC,OC,KH,KW,SH,SW',
                 model_names[2]: 'CNN,COST,T_COST,SD_COST,PDIM,PRATIO,' \
                                    + 'IH,IW,IC,OC,KH,KW,SH,SW',
                 model_names[3]: 'CNN,COST,T_COST,SD_COST,PDIM,PRATIO,' \
                                    + 'IH,IW,IC,OC,KH,KW,SH,SW'}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['COST'],
                   model_names[1]: ['COST'],
                   model_names[2]: ['COST'],
                   model_names[3]: ['COST']}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['IH','IW','IC','OC', 'KH', 'KW', 'SH', 'SW']],
                     model_names[1]: [['IH','IW','IC','OC', 'KH', 'KW', 'SH', 'SW']],
                     model_names[2]: [['IH','IW','IC','OC', 'KH', 'KW', 'SH', 'SW']],
                     model_names[3]: [['IH','IW','IC','OC', 'KH', 'KW', 'SH', 'SW']]}
    return feature_names

class MLFullyConnectedDatasetInfo(DatasetInfo):
  def model_names(self):
    model_names = ['t_fc_cpu_gemv',
                   't_fc_gpu_direct']
    return model_names

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,COMP,T_COMP,SD_COMP,PDIM,PRATIO,' \
                                    + 'M,K,MB,KB,MAX_MB_THR',
                 model_names[1]: ('CNN,COMP,T_{},T_{}_BLK,SD_COMP,PDIM,PRATIO,' \
                                    + 'IH,IW,IWB_SIZE,IC,ICB,OC,OCB,N_{}').format(GPU_ATOM, GPU_ATOM, GPU_ATOM)}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_COMP'],
                   model_names[1]: ['T_{}'.format(GPU_ATOM)]}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['M','K']],
                     model_names[1]: [['IH','IW','IC','OCB']]}
    return feature_names

class MLFullyConnectedOpDatasetInfo(DatasetInfo):
  def model_names(self):
    return ['t_fc_op_cpu',
            't_fc_op_gpu']

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,T_OP,SD_T_OP,PDIM,PRATIO,' \
                                 + 'IH,IW,IC,OC,KH,KW,FLOPS,PARAMS',
                 model_names[1]: 'CNN,T_OP,SD_T_OP,PDIM,PRATIO,' \
                                 + 'IH,IW,IC,OC,KH,KW,FLOPS,PARAMS'}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_OP'],
                   model_names[1]: ['T_OP']}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'FLOPS', 'PARAMS']],
                     model_names[1]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'FLOPS', 'PARAMS']]}
    return feature_names

class MLDeconv2dDatasetInfo(DatasetInfo):
  def model_names(self):
    model_names = ['t_deconv2d_cpu_direct',
                   't_deconv2d_gpu_direct']
    return model_names

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,COMP,T_COMP,SD_COMP,' \
                                    + 'IH,IW,IC,OC,KH,KW,SH,SW,MAX_TILE_THR,FLOPS_PER_TILE',
                 model_names[1]: ('CNN,COMP,T_{},T_{}_ICB,SD_COMP,' \
                                    + 'IH,IW,IC,ICB,OH,OW,OWB,OC,OCB,K,S,N_{}').format(GPU_ATOM, GPU_ATOM, GPU_ATOM)}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_COMP'],
                   model_names[1]: ['T_{}_ICB'.format(GPU_ATOM)]}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['IH', 'IW', 'IC', 'OC', 'KH', 'KW', 'SH', 'SW']],
                     model_names[1]: [['IH', 'IW', 'ICB', 'OCB', 'K', 'S']]}
    return feature_names

class MLMatMulDatasetInfo(DatasetInfo):
  def model_names(self):
    model_names = ['t_matmul_cpu_gemm',
                   't_matmul_gpu_direct']
    return model_names

  def col_names(self):
    model_names = self.model_names()
    col_names = {model_names[0]: 'CNN,PACK_LB,PACK_RB,COMP,T_PACK_LB,T_PACK_RB,T_COMP,SD_PACK_LB,SD_PACK_RB,SD_COMP,' \
                                    + 'M,K,N,MB,KB,NB,MAX_MB_THR,MAX_NB_THR',
                 model_names[1]: ('CNN,COMP,T_{},T_{}_BLK,SD_COMP,' \
                                    + 'M,K,N,MB,KB,NB,N_{}').format(GPU_ATOM, GPU_ATOM, GPU_ATOM)}
    return col_names

  def label_names(self):
    model_names = self.model_names()
    label_names = {model_names[0]: ['T_PACK_LB', 'T_PACK_RB', 'T_COMP'],
                   model_names[1]: ['T_{}_BLK'.format(GPU_ATOM)]}
    return label_names

  def feature_names(self):
    model_names = self.model_names()
    feature_names = {model_names[0]: [['M', 'K'], ['K', 'N'], ['M', 'K', 'N']],
                     model_names[1]: [['M', 'K', 'N']]}
    return feature_names
