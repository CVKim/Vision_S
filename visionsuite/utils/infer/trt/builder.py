import logging
import os
import sys

import tensorrt as trt

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class TrtBuilder:

    def __init__(self, workspace=2):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        # load_tensorrt_plugin()
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)
        profile = self.builder.create_optimization_profile()

        # input_shapes = {'input': {'min_shape': [1, 3, 320, 320], 'opt_shape': [1, 3, 1024, 1024], 'max_shape': [2, 3, 1024, 1024]}}
        # for input_name, param in input_shapes.items():
        #     min_shape = param['min_shape']
        #     opt_shape = param['opt_shape']
        #     max_shape = param['max_shape']
        #     profile.set_shape(input_name, min_shape, opt_shape, max_shape)

        self.config.add_optimization_profile(profile)
        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        print("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        # assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size

    def create_engine(self,
                      engine_path,
                      precision,
                      calib_input=None,
                      calib_cache=None,
                      calib_num_images=5000,
                      calib_batch_size=8):

        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        # inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(bytearray(engine.serialize()))

if __name__ == '__main__':
    
    onnx_model_path = '/DeepLearning/etc/_athena_tests/benchmark/talos/python/deeplabv3plus/export/tenneco_outer.onnx'
    trt_path = '/DeepLearning/etc/_athena_tests/benchmark/talos/python/deeplabv3plus/export/tenneco_outer.trt'
    precision = 'fp32'
    
    builder = TrtBuilder()
    builder.create_network(onnx_model_path)
    builder.create_engine(trt_path, precision)