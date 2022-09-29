# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 1
DEFAULT_PORT = 6010

MODEL='66B'
# MODEL='1.3B'

if MODEL == "66B":
    MODEL_PARALLEL = 8
    TOTAL_WORLD_SIZE = 8
else:
    MODEL_PARALLEL = 2
    TOTAL_WORLD_SIZE = 2
MAX_BEAM = 16

try:
    # internal logic denoting where checkpoints are in meta infrastructure
    from metaseq_internal.constants import MODEL_SHARED_FOLDER, LOCAL_SSD
except ImportError:
    # CHECKPOINT_FOLDER should point to a shared drive (e.g. NFS) where the
    # checkpoints from S3 are stored. As an example:
    # CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
    # $ ls /example/175B/reshard_no_os
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"


# where to find the raw files on nfs
# CHECKPOINT_FOLDER = os.path.join(MODEL_SHARED_FOLDER, "175B", "consolidated_mp_8")

# # # where to store them on SSD for faster loading
# CHECKPOINT_LOCAL_FOLDER = os.path.join(LOCAL_SSD, "175B", "consolidated_mp_8")

# CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_LOCAL_FOLDER, "consolidated.pt")

# HAO CHECKPOINT_FOLDER = os.path.join(MODEL_SHARED_FOLDER, MODEL, f"consolidated_mp_{MODEL_PARALLEL}")
CHECKPOINT_FOLDER = "/nvme0/haoyu/metaseq-opt-models/66b/dependencies"

# where to store them on SSD for faster loading
# HAO CHECKPOINT_LOCAL_FOLDER = os.path.join(LOCAL_SSD, MODEL, f"consolidated_mp_{MODEL_PARALLEL}")

# HAO CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_LOCAL_FOLDER, "consolidated.pt")

# tokenizer files
# HAO BPE_MERGES = os.path.join("/data/gpt-z/models/gptz/", "gpt2-merges.txt")
# HAO BPE_VOCAB = os.path.join("/data/gpt-z/models/gptz/", "gpt2-vocab.json")
BPE_MERGES = os.path.join(CHECKPOINT_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(CHECKPOINT_FOLDER, "gpt2-vocab.json")
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "reshard.pt")


LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--ddp-backend pytorch_ddp",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    # f"--path {CHECKPOINT_LOCAL}",
    f"--path {MODEL_FILE}",
    "--beam 1 --nbest 1",
    # "--distributed-port 13000",
    "--distributed-port -1",     # HAO changed from 13000 to -1
    "--checkpoint-shard-count 1",
    # "--use-sharded-state",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    f"{CHECKPOINT_FOLDER}",  # required "data" argument.
]
