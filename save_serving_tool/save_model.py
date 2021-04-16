import argparse
import os
import shutil

import oneflow as flow
import oneflow.typing as tp

from resnet50_model import resnet50

def _init_oneflow_env_and_config():
    flow.env.init()
    flow.enable_eager_execution(False)
    flow.config.enable_legacy_model_io(True)

def _make_resnet50_predict_func(args):
    batch_size = 1
    channels = 3

    func_cfg = flow.function_config()
    func_cfg.default_placement_scope(flow.scope.placement("cpu", "0:0"))

    @flow.global_function("predict", function_config=func_cfg)
    def predict_fn(
        images: tp.Numpy.Placeholder((1, args.image_height, args.image_width, channels), dtype=flow.float)
    ) -> tp.Numpy:
        logits = resnet50(images, args, training=False)
        predictions = flow.nn.softmax(logits)
        return predictions

    return predict_fn


def main(args):
    _init_oneflow_env_and_config()

    predict_fn = _make_resnet50_predict_func(args)
    flow.train.CheckPoint().load(args.model_dir)
    print("predict_fn construct finished")

    saved_model_path = args.save_dir
    model_version = args.model_version

    model_version_path = os.path.join(saved_model_path, str(model_version))
    if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
        if args.force_save:
            print(
                f"WARNING: The model version path '{model_version_path}' already exist"
                ", old version directory will be replaced"
            )
            shutil.rmtree(model_version_path)
        else:
            raise ValueError(
                f"The model version path '{model_version_path}' already exist"
            )

    saved_model_builder = (
        flow.saved_model.ModelBuilder(saved_model_path)
        .ModelName(args.model_name)
        .Version(model_version)
    )
    saved_model_builder.AddFunction(predict_fn).Finish()
    saved_model_builder.Save()


def _parse_args():
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("flags for save resnet50 model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="resnet50_nhwc",
        help="model parameters directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="resnet50_models",
        help="directory to save models",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet50", help="model name"
    )
    parser.add_argument("--model_version", type=int, default=1, help="model version")
    parser.add_argument(
        "--force_save",
        default=False,
        action="store_true",
        help="force save model whether already exists or not",
    )
    parser.add_argument(
        "--image_width", type=int, default=224, help="input image width"
    )
    parser.add_argument(
        "--image_height", type=int, default=224, help="input image height"
    )
    parser.add_argument(
        "--channel_last",
        type=str2bool,
        default=True,
        help="Whether to use use channel last mode(nhwc)",
    )
    # fuse bn relu or bn add relu
    parser.add_argument(
        "--fuse_bn_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--fuse_bn_add_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization add relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--pad_output",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to pad the output to number of image channels to 4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
