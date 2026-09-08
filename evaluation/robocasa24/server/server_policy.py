import argparse
import logging
import os
from pathlib import Path

import torch

from ace_ego_0.model.framework.base_framework import baseframework
from evaluation.robocasa24.server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    os.environ["ACE_EGO_0_GR1_URDF"] = str(args.urdf_path)
    vla = baseframework.from_pretrained(
        args.ckpt_path,
        framework_override=args.framework_override,
        allow_missing_norm_stats=True,
    )

    if args.use_bf16:
        logging.info("Converting model to bfloat16...")
        vla = vla.to(torch.bfloat16)

    logging.info("Moving ACE-Ego-0 to CUDA (this may take a while)...")
    vla = vla.to("cuda")
    logging.info("Model moved to CUDA successfully")
    vla = vla.eval()

    server = WebsocketPolicyServer(
        policy=vla,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"model": "ACE-Ego-0", "benchmark": "GR1 RoboCasa 24"},
    )
    logging.info("Policy server listening on %s:%s", args.host, args.port)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    default_urdf = Path(__file__).resolve().parents[3] / "assets" / "GR1T2_with_hands.urdf"
    parser.add_argument("--urdf", dest="urdf_path", type=str, default=str(default_urdf))
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--framework_override",
        type=str,
        default=None,
        help="Override the framework name from the checkpoint config.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
