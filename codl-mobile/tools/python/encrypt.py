# Copyright 2019 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import hashlib
from jinja2 import Environment, FileSystemLoader
from py_proto import mace_pb2
from utils import device
from utils import util
from utils.util import mace_check
from utils.util import MaceLogger
from utils import config_parser
from utils.config_parser import CPP_KEYWORDS
from utils.config_parser import ModelKeys

GENERATED_NAME = set()


def generate_obfuscated_name(namespace, name):
    md5 = hashlib.md5()
    md5.update(namespace.encode("utf-8"))
    md5.update(name.encode("utf-8"))
    md5_digest = md5.hexdigest()

    name = md5_digest[:8]
    while name in GENERATED_NAME:
        name = md5_digest
        assert name not in GENERATED_NAME
    GENERATED_NAME.add(name)
    return name


def generate_tensor_map(tensors):
    tensor_map = {}
    for t in tensors:
        if t.name not in tensor_map:
            tensor_map[t.name] = generate_obfuscated_name("tensor", t.name)
    return tensor_map


def generate_in_out_map(ops, tensor_map):
    in_out_map = {}
    for op in ops:
        op.name = generate_obfuscated_name("op", op.name)
        for input_name in op.input:
            if input_name not in in_out_map:
                if input_name in tensor_map:
                    in_out_map[input_name] = tensor_map[input_name]
                else:
                    in_out_map[input_name] = generate_obfuscated_name(
                        "in", input_name)
        for output_name in op.output:
            if output_name not in in_out_map:
                if output_name in tensor_map:
                    in_out_map[output_name] = tensor_map[output_name]
                else:
                    in_out_map[output_name] = generate_obfuscated_name(
                        "out", output_name)
    return in_out_map


def stringfy(value):
    return ', '.join('"{0}"'.format(w) for w in value)


def obfuscate_name(model):
    input_nodes = set()
    for input_node in model.input_info:
        input_nodes.add(input_node.name)
    output_nodes = set()
    for output_node in model.output_info:
        output_nodes.add(output_node.name)
    tensor_map = generate_tensor_map(model.tensors)
    in_out_map = generate_in_out_map(model.op, tensor_map)
    for t in model.tensors:
        if t.name not in input_nodes and t.name not in output_nodes:
            t.name = tensor_map[t.name]
    for op in model.op:
        for i in range(len(op.input)):
            if op.input[i] not in input_nodes:
                op.input[i] = in_out_map[op.input[i]]
        for i in range(len(op.output)):
            if op.output[i] not in output_nodes:
                op.output[i] = in_out_map[op.output[i]]


def save_model_to_code(namespace, model, params, model_checksum,
                       params_checksum, device, output, gencode_params):
    util.mkdir_p(output)
    cwd = os.path.dirname(__file__)
    j2_env = Environment(
        loader=FileSystemLoader(cwd + "/template"), trim_blocks=True)
    j2_env.filters["stringfy"] = stringfy

    template_name = "tensor_source.jinja2"
    counter = 0
    for tensor in model.tensors:
        # convert tensor
        source = j2_env.get_template(template_name).render(
            tensor=tensor,
            tensor_id=counter,
            tag=namespace,
        )
        with open(output + "/tensor" + str(counter) + ".cc", "w") as f:
            f.write(source)
        counter += 1

    if gencode_params:
        template_name = "tensor_data.jinja2"
        source = j2_env.get_template(template_name).render(
            tag=namespace,
            model_data_size=len(params),
            model_data=params)
        with open(output + "/tensor_data.cc", "w") as f:
            f.write(source)

    template_name = "operator.jinja2"
    counter = 0
    op_size = len(model.op)

    for start in range(0, op_size, 10):
        source = j2_env.get_template(template_name).render(
            start=start,
            end=min(start + 10, op_size),
            net=model,
            tag=namespace,
            device=device.value,
        )
        with open(output + "/op" + str(counter) + ".cc", "w") as f:
            f.write(source)
        counter += 1

    # generate model source files
    build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template_name = "model.jinja2"
    checksum = "{},{}".format(model_checksum, params_checksum)
    source = j2_env.get_template(template_name).render(
        net=model,
        tag=namespace,
        checksum=checksum,
        build_time=build_time)
    with open(output + "/model.cc", "w") as f:
        f.write(source)

    template_name = 'model_header.jinja2'
    source = j2_env.get_template(template_name).render(tag=namespace, )
    with open(output + "/" + namespace + '.h', "w") as f:
        f.write(source)


def save_model_to_file(model_name, model, params, output):
    util.mkdir_p(output)
    with open(output + "/" + model_name + ".pb", "wb") as f:
        f.write(model.SerializeToString())
    with open(output + "/" + model_name + ".data", "wb") as f:
        f.write(params)


def encrypt(model_name, model_file, params_file, device, output,
            is_obfuscate=False, gencode_model=False, gencode_params=False):
    model_checksum = util.file_checksum(model_file)
    params_checksum = util.file_checksum(params_file)

    with open(model_file, "rb") as model_file:
        with open(params_file, "rb") as params_file:
            model = mace_pb2.NetDef()
            model.ParseFromString(model_file.read())
            params = bytearray(params_file.read())

            if is_obfuscate:
                obfuscate_name(model)
            save_model_to_file(model_name, model, params, output)
            if gencode_model:
                save_model_to_code(model_name, model, params, model_checksum,
                                   params_checksum, device, output + "/code/",
                                   gencode_params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        help="the namespace of gernerated code")
    parser.add_argument(
        '--model_file',
        type=str,
        help="model file")
    parser.add_argument(
        '--params_file',
        type=str,
        help="params file")
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="cpu/gpu/hexagon/hta/apu")
    parser.add_argument(
        '--config',
        type=str,
        help="model config")
    parser.add_argument(
        "--no_obfuscate",
        action="store_true",
        help="obfuscate model names")
    parser.add_argument(
        "--gencode_model",
        action="store_true",
        help="generate model code")
    parser.add_argument(
        "--gencode_param",
        action="store_true",
        help="generate params code")
    parser.add_argument(
        '--output',
        type=str,
        default="build",
        help="output dir")

    flgs, _ = parser.parse_known_args()
    mace_check(flgs.model_name not in CPP_KEYWORDS, "model name cannot be cpp"
                                                    "keywords")
    return flgs


def gen_mace_engine_factory(model_name, embed_model_data, output):
    util.mkdir_p(output)
    cwd = os.path.dirname(__file__)
    j2_env = Environment(
        loader=FileSystemLoader(cwd + "/template"), trim_blocks=True)
    # generate mace_run BUILD file
    template_name = 'mace_engine_factory.h.jinja2'
    model_name = list(model_name)
    source = j2_env.get_template(template_name).render(
        model_tags=model_name,
        embed_model_data=embed_model_data,
    )
    with open(output + '/mace_engine_factory.h', "w") as f:
        f.write(source)


if __name__ == '__main__':
    flags = parse_args()
    codegen_dir = "mace/codegen/models"
    device.execute("rm -rf %s/*" % codegen_dir)

    models = []
    if flags.config:
        conf = config_parser.parse(flags.config)

        for name, model_conf in conf["models"].items():
            model_conf = config_parser.normalize_model_config(model_conf)
            if not flags.model_name or name == flags.model_name:
                MaceLogger.info("Encrypt model %s" % name)
                encrypt(name,
                        "build/%s/model/%s.pb" % (name, name),
                        "build/%s/model/%s.data" % (name, name),
                        model_conf[ModelKeys.runtime],
                        codegen_dir + "/" + name,
                        not flags.no_obfuscate,
                        flags.gencode_model,
                        flags.gencode_param)
                models.append(name)
                os.rename("%s/%s/%s.pb" % (codegen_dir, name, name),
                          "build/%s/model/%s.pb" % (name, name))
                os.rename("%s/%s/%s.data" % (codegen_dir, name, name),
                          "build/%s/model/%s.data" % (name, name))
    else:
        device_type = config_parser.parse_device_type(flags.device)
        encrypt(flags.model_name, flags.model_file, flags.params_file,
                device_type, codegen_dir, not flags.no_obfuscate,
                flags.gencode_model, flags.gencode_param)
        models.append(flags.model_name)
        os.rename(
            "%s/%s/%s.pb" % (codegen_dir, flags.model_name, flags.model_name),
            "build/%s/model/%s.pb" % (flags.model_name, flags.model_name))
        os.rename(
            "%s/%s/%s.data" % (codegen_dir, flags.model_name,
                               flags.model_name),
            "build/%s/model/%s.data" % (flags.model_name, flags.model_name))

    if flags.gencode_model:
        gen_mace_engine_factory(models, flags.gencode_param,
                                "mace/codegen/engine")
