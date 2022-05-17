
from utils.common.basic_type import MemoryObject
from utils.op.conv2d_common_calc import *
from utils.op.op_adb_utils import *

PDIM_HEIGHT      = [PartitionDimension.HEIGHT]
PDIM_OUT_CHANNEL = [PartitionDimension.OUT_CHANNEL]
PDIM_ALL         = [PartitionDimension.HEIGHT, PartitionDimension.OUT_CHANNEL]

def measure_conv2d_on_device(conv2d_param,
                             dev_name,
                             pdim=PartitionDimension.HEIGHT,
                             pratio=1.0,
                             do_data_transform=False,
                             do_compute=True,
                             gpu_mem_object=MemoryObject.GPU_IMAGE):
    lat_list = OpAdbUtils.run_op_on_device('Conv2D',
                                           conv2d_param,
                                           dev_name,
                                           'none',
                                           pdim,
                                           pratio,
                                           do_data_transform,
                                           do_compute,
                                           gpu_mem_object)
    return lat_list[0]
