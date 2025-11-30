import argparse
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


_DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".yaml")


def _resolve_config_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_file():
        return candidate.resolve()
    alt_candidate = (_DEFAULT_CONFIG_PATH.parent / candidate).resolve()
    if alt_candidate.is_file():
        return alt_candidate
    raise FileNotFoundError(f"Config file not found: {path_str}")


def _load_defaults(config_path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"Config at {config_path} is not a mapping.")
    return container  # type: ignore[return-value]


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).lower()
    if value_str in {"true", "1", "yes", "y", "t"}:
        return True
    if value_str in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _build_parser(defaults: Dict[str, Any], config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyper-parameters management")

    parser.add_argument(
        "--config_path",
        default=str(config_path),
        help="Path to YAML config file",
    )

    # Hardware options
    parser.add_argument("--n_threads", type=int, default=defaults["n_threads"], help="number of threads for data loading")
    parser.add_argument("--cpu", action="store_true", default=defaults["cpu"], help="use cpu only")
    parser.add_argument("--gpu_id", type=int, default=defaults["gpu_id"], help="use gpu only")
    parser.add_argument("--local_rank", type=int, default=defaults["local_rank"])

    parser.add_argument("--resize_radio", type=float, default=defaults["resize_radio"])
    parser.add_argument("--r_resize", type=float, default=defaults["r_resize"])
    parser.add_argument("--device_id", default=defaults["device_id"], type=str)

    # Model config
    parser.add_argument("--sam2_pretrain", default=defaults["sam2_pretrain"], help="path to sam2 weights")
    parser.add_argument("--adadim", type=int, default=defaults["adadim"], help="dims for adapter in sam2")
    parser.add_argument("--rfbdim", type=int, default=defaults["rfbdim"], help="dim for rfb block")

    # Dataset parameters
    parser.add_argument("--dataset_name", default=defaults["dataset_name"], help="datasets name")
    parser.add_argument("--dataset_img_path", default=defaults["dataset_img_path"], help="Train datasets image root path")
    parser.add_argument("--dataset_img_test_path", default=defaults["dataset_img_test_path"], help="Train datasets label root path")
    parser.add_argument("--test_data_path", default=defaults["test_data_path"], help="Test datasets root path")
    parser.add_argument("--predict_seed_path", default=defaults["predict_seed_path"], help="Seed root path")
    parser.add_argument("--predict_centerline_path", default=defaults["predict_centerline_path"], help="Saved centerline result root path")
    parser.add_argument("--predict_swc_path", default=defaults["predict_swc_path"], help="Saved swc result root path")

    # Training patch / stride
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="batch size of trainset")
    parser.add_argument("--valid_rate", type=float, default=defaults["valid_rate"])
    parser.add_argument(
        "--data_shape",
        type=int,
        nargs=3,
        default=defaults["data_shape"],
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help="input data shape",
    )

    parser.add_argument("--test_patch_height", type=int, default=defaults["test_patch_height"])
    parser.add_argument("--test_patch_width", type=int, default=defaults["test_patch_width"])
    parser.add_argument("--test_patch_depth", type=int, default=defaults["test_patch_depth"])
    parser.add_argument("--stride_height", type=int, default=defaults["stride_height"])
    parser.add_argument("--stride_width", type=int, default=defaults["stride_width"])
    parser.add_argument("--stride_depth", type=int, default=defaults["stride_depth"])

    # IO paths
    parser.add_argument("--model_save_dir", default=defaults["model_save_dir"], help="save path of trained model")
    parser.add_argument("--log_save_dir", default=defaults["log_save_dir"], help="save path of trained log")

    # Train
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=defaults["lr"], metavar="LR", help="learning rate")
    parser.add_argument("--vector_bins", type=int, default=defaults["vector_bins"])
    parser.add_argument("--train_seg", type=_str2bool, default=defaults["train_seg"])

    # Test / inference
    parser.add_argument("--print_info", type=_str2bool, default=defaults["print_info"])
    parser.add_argument(
        "--tracing_strategy_mode",
        default=defaults["tracing_strategy_mode"],
        type=str,
        help="centerline | angle | anglecenterlined",
    )
    parser.add_argument("--train_or_test", default=defaults["train_or_test"])
    parser.add_argument("--to_restore", type=_str2bool, default=defaults["to_restore"])

    return parser


def _parse_args() -> argparse.Namespace:
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config_path", default=str(_DEFAULT_CONFIG_PATH))
    known_args, _ = initial_parser.parse_known_args()
    resolved_config_path = _resolve_config_path(known_args.config_path)
    defaults = _load_defaults(resolved_config_path)
    parser = _build_parser(defaults, resolved_config_path)
    args = parser.parse_args()
    args.config_path = str(_resolve_config_path(args.config_path))
    if isinstance(args.data_shape, tuple):
        args.data_shape = list(args.data_shape)
    elif isinstance(args.data_shape, List):
        args.data_shape = list(args.data_shape)
    return args


args = _parse_args()
