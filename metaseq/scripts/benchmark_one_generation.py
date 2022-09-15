#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq_cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""
import sys
import os
import random

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    # CHECKPOINT_LOCAL,
    # CHECKPOINT_FOLDER,
    LAUNCH_ARGS,
)
from metaseq.service.utils import encode_fn, build_logger
import torch.profiler as profiler

port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()

@torch.no_grad()
def worker_main(cfg: MetaseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841
    logger.info("loaded model")

    prompts = [
        encode_fn(generator, x) for x in
        [
            "this is a test",
            "this is another test",
            # "hi, how are you?",
            "they dont really care about us",
            # "another one bites the dust",
        ]
    ]
    logger.info("forwarded token")
    model = models[0].decoder
    model.eval()

    # import pdb; pdb.set_trace()
    request_object = {
        "inputs": prompts,
        "min_tokens": [4],
        "max_tokens": [4],
        "temperature": 1.0,
        "top_p": 0.0,
        "n": 1,
        "seed": 11619
    }
    generations = generator.generate(**request_object)
    logger.info('done generation')
    logger.info(generations[0][0]['text'])



def cli_main():
    """
    Command line interactive.
    """
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    if len(sys.argv) > 1:
        try:
            from metaseq_internal.constants import TRAINED_MODEL_CONFIG
        except ImportError:
            TRAINED_MODEL_CONFIG = {}
        model_name = sys.argv[1]
        env_name = "azure" if len(sys.argv) < 3 else sys.argv[2]
        try:
            model_config = TRAINED_MODEL_CONFIG[env_name][model_name]
        except KeyError:
            print(
                f"{model_name} on {env_name} config not found, please add that in metaseq_internal.constants"
            )
        del model_config["lauch_port"]
        print(model_config)
        LAUNCH_ARGS.update(model_config)
    for k, v in LAUNCH_ARGS.items():
        k = k.replace("_", "-")
        if k != "data":
            flat_launch_args.append(f"--{k}")
        if str(v) != "True":
            flat_launch_args.append(str(v))

    # flat_launch_args += ['--searching']
    gen_method = sys.argv[3]
    if gen_method == 'search': 
        flat_launch_args += ['--searching']
    elif gen_method == 'sample':
        pass
    else:
        print('passing generation argument please, using sampling now')
        
    print("Api launching config")
    print(flat_launch_args)

    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    port = DEFAULT_PORT
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE
    dist_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    cli_main()
