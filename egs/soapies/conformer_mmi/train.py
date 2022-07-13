#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


import argparse
import logging
from pathlib import Path
from shutil import copyfile
from typing import Dict, Optional

import k2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import SoapiesAsrDataModule
from model import TdnnLstm
from lhotse.utils import fix_random_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from icefall.ali import (
    convert_alignments_to_tensor,
    load_alignments,
    lookup_alignments,
)
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    encode_supervisions,
    setup_logger,
    str2bool,
)

#######################################################################
# MatGraph is a thin wrapper around the julia package:
# https://github.com/FAST-ASR/MarkovModels.jl. We use it to
# calculate the LFMMI loss.

import matgraph as mg

#######################################################################


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        conformer_mmi/exp/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    return parser


def get_params(lang) -> AttributeDict:
    params = AttributeDict(
        {
            "graphs_dir": Path(f"data/graphs/{lang}"),
            "exp_dir": Path(f"conformer_mmi/exp/{lang}"),
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 3,
            # parameters for loss
            "beam_size": 6,  # will change it to 8 after some batches (see code)
            "reduction": "sum",
            "use_double_scores": True,
            "num_decoder_layers": 0,
            # parameters for AdamW
            "weight_decay": 5e-4,
            "lr": 1e-3,
            "den_scale": 1.0,
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """
    if params.start_epoch <= 0:
        return

    filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    denfsms,
    numfsms,
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    is_training: bool,
):
    """
    Compute LF-MMI loss given the model and its inputs.

    Args:
      denfsms:
        Batched denominator fsm.
      numfsms:
        Batched numerator fsm.
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.permute(0, 2, 1)
    feature = feature.to(device)

    supervisions = batch["supervisions"]

    # The conversion to numpy is necessary to avoid issue converting
    # the list to a julia array (done my matgraph).
    seqlengths = supervisions["num_frames"].numpy()

    with torch.set_grad_enabled(is_training):
        nnet_output = model(feature)
        # nnet_output is (N, T, C)

        # NOTE: We need `encode_supervisions` to sort sequences with
        # different duration in decreasing order, required by
        # `k2.intersect_dense` called in `LFMMILoss.forward()`
        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=params.subsampling_factor
        )

        loss_fn = mg.LFMMILoss(denfsms, numfsms, params.den_scale)

        mmi_loss = loss_fn(nnet_output, seqlengths)

    loss = mmi_loss

    # train_frames and valid_frames are used for printing.
    if is_training:
        params.train_frames = supervision_segments[:, 2].sum().item()
    else:
        params.valid_frames = supervision_segments[:, 2].sum().item()

    assert loss.requires_grad == is_training

    return loss, mmi_loss.detach()


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    ali: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Run the validation process. The validation loss
    is saved in `params.valid_loss`.
    """
    model.eval()

    tot_loss = 0.0
    tot_mmi_loss = 0.0
    tot_frames = 0.0
    for batch_idx, batch in enumerate(valid_dl):
        loss, mmi_loss = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=False,
            ali=ali,
        )
        assert loss.requires_grad is False
        assert mmi_loss.requires_grad is False

        loss_cpu = loss.detach().cpu().item()
        tot_loss += loss_cpu

        tot_mmi_loss += mmi_loss.detach().cpu().item()

        tot_frames += params.valid_frames

    if world_size > 1:
        s = torch.tensor(
            [tot_loss, tot_mmi_loss, tot_frames],
            device=loss.device,
        )
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        s = s.cpu().tolist()
        tot_loss = s[0]
        tot_mmi_loss = s[1]
        tot_frames = s[2]

    params.valid_loss = tot_loss / tot_frames
    params.valid_mmi_loss = tot_mmi_loss / tot_frames

    if params.valid_loss < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = params.valid_loss


def train_one_epoch(
    denfsm,
    numfsm_paths,
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      denfsm:
        Denominator fsm.
      numfsm_paths:
        Mapping uttid => fsm path for the numerator fsms.
    """
    model.train()

    tot_loss = 0.0  # sum of losses over all batches
    tot_mmi_loss = 0.0

    tot_frames = 0.0  # sum of frames over all batches
    params.tot_loss = 0.0
    params.tot_frames = 0.0
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        numfsms = []
        for cut in batch["supervisions"]["cut"]:
            uttid = cut.supervisions[0].id.split("_sp")[0]
            numfsms.append(mg.FSM.from_files(
                str(numfsm_paths[uttid]),
                str(numfsm_paths[uttid].with_suffix(".smap"))
            ))
        denfsms = mg.BatchFSM.from_list([denfsm for _ in range(len(numfsms))])
        numfsms = mg.BatchFSM.from_list(numfsms)
        if torch.cuda.is_available():
            numfsms = numfsms.cuda()

        loss, mmi_loss = compute_loss(
            denfsms,
            numfsms,
            params=params,
            model=model,
            batch=batch,
            is_training=True,
        )

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        loss_cpu = loss.detach().cpu().item()
        mmi_loss_cpu = mmi_loss.detach().cpu().item()

        tot_frames += params.train_frames
        tot_loss += loss_cpu
        tot_mmi_loss += mmi_loss_cpu

        params.tot_frames += params.train_frames
        params.tot_loss += loss_cpu

        tot_avg_loss = tot_loss / tot_frames
        tot_avg_mmi_loss = tot_mmi_loss / tot_frames

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"batch avg mmi loss {mmi_loss_cpu/params.train_frames:.4f}, "
                f"batch avg loss {loss_cpu/params.train_frames:.4f}, "
                f"total avg mmiloss: {tot_avg_mmi_loss:.4f}, "
                f"total avg loss: {tot_avg_loss:.4f}, "
                f"batch size: {batch_size}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/current_mmi_loss",
                    mmi_loss_cpu / params.train_frames,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/current_loss",
                    loss_cpu / params.train_frames,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/tot_avg_mmi_loss",
                    tot_avg_mmi_loss,
                    params.batch_idx_train,
                )

                tb_writer.add_scalar(
                    "train/tot_avg_loss",
                    tot_avg_loss,
                    params.batch_idx_train,
                )
        if batch_idx > 0 and batch_idx % params.reset_interval == 0:
            tot_loss = 0.0  # sum of losses over all batches
            tot_mmi_loss = 0.0

            tot_frames = 0.0  # sum of frames over all batches

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                ali=valid_ali,
            )
            model.train()
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"valid mmi loss {params.valid_mmi_loss:.4f},"
                f"valid loss {params.valid_loss:.4f},"
                f" best valid loss: {params.best_valid_loss:.4f} "
                f"best valid epoch: {params.best_valid_epoch}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/valid_mmi_loss",
                    params.valid_mmi_loss,
                    params.batch_idx_train,
                )
                tb_writer.add_scalar(
                    "train/valid_loss",
                    params.valid_loss,
                    params.batch_idx_train,
                )

    params.train_loss = params.tot_loss / params.tot_frames

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params(args.lang)
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    #graph_compiler = MmiTrainingGraphCompiler(
    #    params.lang_dir,
    #    device=device,
    #    oov="<UNK>",
    #    sos_id=1,
    #    eos_id=1,
    #)

    logging.info("About to create model")

    with open(params.graphs_dir / "numpdf", "r") as f:
        num_classes = int(f.readline().strip())

    model = TdnnLstm(
        num_features=params.feature_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
    )

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])


    soapies = SoapiesAsrDataModule(args)
    train_dl = soapies.train_dataloaders()
    valid_dl = soapies.valid_dataloaders()

    # Load the numerator fsm mapping.
    numfsm_paths = {}
    for split in ("train", "dev"):
        with open(params["graphs_dir"] / "numfsms" / split / "fsm.scp", "r") as f:
            for line in f:
                uttid, path = line.strip().split()
                numfsm_paths[uttid] = Path(path)

    # Load the denominator graph.
    denfsm = mg.FSM.from_files(
        str(params["graphs_dir"] / "denominator.fsm"),
        str(params["graphs_dir"] / "denominator.smap"),
    )
    if torch.cuda.is_available():
        denfsm = denfsm.cuda()

    for epoch in range(params.start_epoch, params.num_epochs):
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

        cur_lr = optimizer.defaults["lr"]
        if tb_writer is not None:
            tb_writer.add_scalar(
                "train/learning_rate", cur_lr, params.batch_idx_train
            )
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        if rank == 0:
            logging.info("epoch {}, learning rate {}".format(epoch, cur_lr))

        params.cur_epoch = epoch

        train_one_epoch(
            denfsm,
            numfsm_paths,
            params=params,
            model=model,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
        )

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    SoapiesAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()

