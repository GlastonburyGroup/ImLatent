from typing import Literal
from .Science.losses import LOSSID, LOSSID_CLS
from .Engines.WarpDrives import MODELID
from .Engines.WarpDrives.pythaeDrive import PYTHAEMODELID

OPTIMID = {
    0: "Adam",
    1: "AdamW",
    2: "RAdam",
    -1: "DeepSpeedCPUAdam (DeepSpeed): provides 5x to 7x speedup over torch.optim.adam while being used together with deepspeed_stage_2_offload",
    -1: "FusedAdam (DeepSpeed): to be used with deepspeed_stage_3",
    "String": "String to call Optimiser from the torch.hub of pytorch-optimizers"
}

STRATEGIES = {
    "default": "Default Training Strategy: DDP for multi-GPU and/or multi-node training. For single GPU-node training, ignored.",
    "fsdp": "Fully Sharded Data Parallel",
    "ddp_spawn": "Distributed Data Parallel (DDP) with spawn (Not recommended!)",
    "deepspeed": "DeepSpeed: Default settings",
    "deepspeed_stage_1": "DeepSpeed ZeRO Stage 1 (Not recommended!): Shard optimiser states, remains at speed parity with DDP whilst providing memory improvement",
    "deepspeed_stage_2": "DeepSpeed ZeRO Stage 2: Shard optimiser states and gradients, remains at speed parity with DDP whilst providing even more memory improvement",
    "deepspeed_stage_2_offload": "DeepSpeed ZeRO Stage 2 Offload: Offload optimiser states and gradients to CPU. Increases distributed communication volume and GPU-CPU device transfer, but provides significant memory improvement",
    "deepspeed_stage_3": "DeepSpeed ZeRO Stage 3: Shard optimiser states, gradients, parameters and optionally activations. Increases distributed communication volume, but provides even more memory improvement",
    "deepspeed_stage_3_offload": "DeepSpeed ZeRO Stage 3 Offload: Offload optimiser states, gradients, parameters and optionally activations to CPU. Increases distributed communication volume and GPU-CPU device transfer, but even more significant memory improvement."
}