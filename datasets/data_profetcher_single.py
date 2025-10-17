# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
Purpose: initialize a data prefetcher and start preloading batches.
Args:
    loader: a data loader (e.g., PyTorch DataLoader) that yields batches.
    device: target device (e.g., a CUDA device).
"""

import torch


def to_cuda(samples, targets, device):
    # samples = samples.to(device, non_blocking=True)
    samples = {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    # targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    targets = targets.to(device, non_blocking=True)
    return samples, targets


class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    """
    Preload the next batch onto the target device (usually GPU) to improve training throughput.
    Steps:
    - Try to fetch the next batch from the loader.
    - If StopIteration is raised, set next_samples/next_targets to None.
    - Move fetched tensors to the target device using a dedicated CUDA stream.
    """
    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    """
    Return the preloaded batch.
    If prefetching is enabled, wait for the CUDA copy stream, record stream usage
    for tensors, and then trigger preloading of the next batch. Otherwise, fetch
    synchronously from the loader and move to device.
    """
    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                if isinstance(samples, dict):
                    for k, v in samples.items():
                        v.record_stream(torch.cuda.current_stream())
                else:
                    samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                if isinstance(targets, dict):
                    for t in targets:
                        for k, v in t.items():
                            v.record_stream(torch.cuda.current_stream())
                else:
                    targets.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets