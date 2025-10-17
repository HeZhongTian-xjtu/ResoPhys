# %%
import argparse
import logging
import os
import argparse
import logging
import shutil
import datetime
import time

import cv2
import yaml
# from tqdm.notebook import tqdm
from tqdm import tqdm
import sys
import math

from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
from typing import Iterable


# from utils.utils import init_seeds, worker_init_fn, get_rank, uprint, get_device, is_main_process, str2bool, DummyFile, redirect_stdout
import utils.misc as utils
from datasets.data_profetcher_single import data_prefetcher
from utils import setup_logger, get_rank, get_device, str2bool
from loguru import logger
# %%
import h5py
from scipy import interpolate
from datasets.dataset import UBFC_LU_split, COHFACE_LU_split, PURE_LU_split, H5Dataset, ToTensor, Normaliztion
from model import build_net
from loss.loss import ContrastLoss
from utils.signal import cal_hr
from loss.TorchLossComputer import TorchLossComputer

from test_all import evaluate, bland_altman_plot, plot_ppg
# from test_all import evaluate_split_eq
# from test_all import bland_altman_plot, plot_ppg
from utils.signal import butter_bandpass
@torch.no_grad()
def evaluate_split_eq(args, val_data, ckpt_path, plot=False, save_dir="./"):
    # self.net = None
    torch.cuda.empty_cache()
    # Split a video into three equal parts
    device = get_device()
    state_dict = torch.load(ckpt_path, map_location=device)["model"]
    dataloader = val_data
    n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader)

    loop = tqdm(enumerate(dataloader,start=1),
                total=n,
                file=sys.stdout,
                desc="val",
                leave=True,
                )

    # for cal metric
    clip_hr_pred = []
    clip_hr_gt = []
    video_clip_pred = []
    video_clip_gt = []
    # for plot
    select_visual = set(np.random.randint(0, len(dataloader), 5))
    psd_pred_f_list = []
    psd_pred_list = []
    psd_gt_f_list = []
    psd_gt_list = []
    pred_list = []
    gt_list = []
    file_name_list = []
    for step, batch in loop:
        # assert len(batch)==1
        video = batch["video"] # [B,C,T,H,W]
        video = video.to(device); ecg = batch["ecg"].to(device); clip_avg_hr = batch["clip_avg_hr"].to(device)

        ecg = batch["ecg"]
        if "seg" in batch.keys():
            mask = batch["seg"]
        location = batch["location"][0]
        if video.shape[2] < 300:
            continue
        time_interval = int(video.shape[2]//3)  # split into three equal parts
        args.T = time_interval
        net = build_net(args, device)  # rebuild network according to the temporal length
        net.load_state_dict(state_dict)
        net = net.to(device)
        net.eval()
        num_blocks = 3
        _video_pred = []
        _video_gt = []
        for b in range(num_blocks):
            video_clip = video[:, :, b*time_interval:(b+1)*time_interval, :, :].to(device)
            ecg_clip = ecg[:, b*time_interval:(b+1)*time_interval].to(device)
            input_dict = {"video":video_clip}
            if "seg" in batch.keys():
                mask_clip = mask[:, b*time_interval:(b+1)*time_interval].to(device)
                input_dict["seg"] = mask_clip
            model_outputs = net(input_dict)#, "seg":mask_clip})
            rppg = model_outputs["rppg"]

            # normalize
            rppg = (rppg - rppg.min()) / (rppg.max() - rppg.min())
            ecg_clip = (ecg_clip - ecg_clip.min()) / (ecg_clip.max() - ecg_clip.min())
            rppg = butter_bandpass(rppg.cpu().numpy(), 0.6, 4,30)
            ecg_clip = butter_bandpass(ecg_clip.cpu().numpy(), 0.6, 4,30)
            rppg = torch.from_numpy(rppg.copy()).to(device)
            ecg_clip = torch.from_numpy(ecg_clip.copy()).to(device)
            hr_pred, _psd_pred_f, _psd_pred = cal_hr(rppg)
            hr_gt, _psd_gt_f, _psd_gt  = cal_hr(ecg_clip)
            clip_hr_pred.append(hr_pred.item())
            clip_hr_gt.append(hr_gt.item())
            _video_pred.append(hr_pred.item())
            _video_gt.append(hr_gt.item())

            # for plot
            _psd_pred_f = _psd_pred_f.cpu().numpy().reshape(-1); psd_pred_f_list.append(_psd_pred_f)
            _psd_pred = _psd_pred.cpu().numpy().reshape(-1); psd_pred_list.append(_psd_pred)
            _psd_gt_f = _psd_gt_f.cpu().numpy().reshape(-1); psd_gt_f_list.append(_psd_gt_f)
            _psd_gt = _psd_gt.cpu().numpy().reshape(-1); psd_gt_list.append(_psd_gt)
            rppg = rppg.cpu().numpy().reshape(-1); pred_list.append(rppg)
            ecg_clip = ecg_clip.cpu().numpy().reshape(-1); gt_list.append(ecg_clip)

            if step in select_visual and b==2 and plot:
                plot_ppg(rppg, ecg_clip, _psd_pred_f, _psd_pred, _psd_gt_f, _psd_gt, location, save_dir)

        video_clip_pred.append(np.mean(_video_pred))
        video_clip_gt.append(np.mean(_video_gt))
        del net
        torch.cuda.empty_cache()
        # if step > 30:
        #     break
    # Compute metrics
    video_clip_pred_np = np.array(video_clip_pred)
    video_clip_gt_np = np.array(video_clip_gt)
    clip_hr_pred_np = np.array(clip_hr_pred)
    clip_hr_gt_np = np.array(clip_hr_gt)

    video_mae = np.mean(np.abs(video_clip_pred_np-video_clip_gt_np))
    video_rmse = np.sqrt(np.mean((video_clip_pred_np-video_clip_gt_np)**2))
    clip_mae = np.mean(np.abs(clip_hr_pred_np-clip_hr_gt_np))
    clip_rmse = np.sqrt(np.mean((clip_hr_pred_np-clip_hr_gt_np)**2))
    video_r = np.corrcoef(video_clip_pred_np, video_clip_gt_np)[0,1]
    step_log = {"video_mae":video_mae, "video_rmse":video_rmse, "clip_mae":clip_mae, "clip_rmse":clip_rmse, "video_r":video_r}

    bland_altman_plot(video_clip_gt_np, video_clip_pred_np, save_dir)
    return_dict = {"mae": video_mae, "rmse": video_rmse, "r": video_r, "step_log": step_log}
    return return_dict


@torch.no_grad()
def evaluate_clip(args, val_data, ckpt_path, plot=False, save_dir="./", time_interval=160):
    # self.net = None
    torch.cuda.empty_cache()
    # Split a video into three equal parts
    device = get_device()
    state_dict = torch.load(ckpt_path, map_location=device)["model"]
    dataloader = val_data
    n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader)

    loop = tqdm(enumerate(dataloader,start=1),
                total=n,
                file=sys.stdout,
                desc="val",
                leave=True,
                )
    args.T = time_interval
    net = build_net(args, device) # rebuild network according to the temporal length
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    # for calculating metrics
    clip_hr_pred = []
    clip_hr_gt = []
    video_clip_pred = []
    video_clip_gt = []
    # for plotting
    select_visual = set(np.random.randint(0, len(dataloader), 5))
    psd_pred_f_list = []
    psd_pred_list = []
    psd_gt_f_list = []
    psd_gt_list = []
    pred_list = []
    gt_list = []
    file_name_list = []
    for step, batch in loop:
        # assert len(batch)==1
        video = batch["video"] # [B,C,T,H,W]
        video = video.to(device); ecg = batch["ecg"].to(device); clip_avg_hr = batch["clip_avg_hr"].to(device)

        ecg = batch["ecg"]
        if "seg" in batch.keys():
            mask = batch["seg"]
        location = batch["location"][0]
        if video.shape[2] < 300:
            continue
        num_blocks = 3
        _video_pred = []
        _video_gt = []
        for b in range(num_blocks):
            video_clip = video[:, :, b*time_interval:(b+1)*time_interval, :, :].to(device)
            ecg_clip = ecg[:, b*time_interval:(b+1)*time_interval].to(device)
            input_dict = {"video":video_clip}
            if "seg" in batch.keys():
                mask_clip = mask[:, b*time_interval:(b+1)*time_interval].to(device)
                input_dict["seg"] = mask_clip
            model_outputs = net(input_dict)#, "seg":mask_clip})
            rppg = model_outputs["rppg"]

            # normalize
            rppg = (rppg - rppg.min()) / (rppg.max() - rppg.min())
            ecg_clip = (ecg_clip - ecg_clip.min()) / (ecg_clip.max() - ecg_clip.min())
            rppg = butter_bandpass(rppg.cpu().numpy(), 0.6, 4,30)
            ecg_clip = butter_bandpass(ecg_clip.cpu().numpy(), 0.6, 4,30)
            rppg = torch.from_numpy(rppg.copy()).to(device)
            ecg_clip = torch.from_numpy(ecg_clip.copy()).to(device)
            hr_pred, _psd_pred_f, _psd_pred = cal_hr(rppg)
            hr_gt, _psd_gt_f, _psd_gt  = cal_hr(ecg_clip)
            clip_hr_pred.append(hr_pred.item())
            clip_hr_gt.append(hr_gt.item())
            _video_pred.append(hr_pred.item())
            _video_gt.append(hr_gt.item())

            # for plot
            _psd_pred_f = _psd_pred_f.cpu().numpy().reshape(-1); psd_pred_f_list.append(_psd_pred_f)
            _psd_pred = _psd_pred.cpu().numpy().reshape(-1); psd_pred_list.append(_psd_pred)
            _psd_gt_f = _psd_gt_f.cpu().numpy().reshape(-1); psd_gt_f_list.append(_psd_gt_f)
            _psd_gt = _psd_gt.cpu().numpy().reshape(-1); psd_gt_list.append(_psd_gt)
            rppg = rppg.cpu().numpy().reshape(-1); pred_list.append(rppg)
            ecg_clip = ecg_clip.cpu().numpy().reshape(-1); gt_list.append(ecg_clip)

            if step in select_visual and b==2 and plot:
                plot_ppg(rppg, ecg_clip, _psd_pred_f, _psd_pred, _psd_gt_f, _psd_gt, location, save_dir)

        video_clip_pred.append(np.mean(_video_pred))
        video_clip_gt.append(np.mean(_video_gt))
        # if step > 30:
        #     break
    # Compute metrics
    video_clip_pred_np = np.array(video_clip_pred)
    video_clip_gt_np = np.array(video_clip_gt)
    clip_hr_pred_np = np.array(clip_hr_pred)
    clip_hr_gt_np = np.array(clip_hr_gt)

    video_mae = np.mean(np.abs(video_clip_pred_np-video_clip_gt_np))
    video_rmse = np.sqrt(np.mean((video_clip_pred_np-video_clip_gt_np)**2))
    clip_mae = np.mean(np.abs(clip_hr_pred_np-clip_hr_gt_np))
    clip_rmse = np.sqrt(np.mean((clip_hr_pred_np-clip_hr_gt_np)**2))
    video_r = np.corrcoef(video_clip_pred_np, video_clip_gt_np)[0,1]
    step_log = {"video_mae":video_mae, "video_rmse":video_rmse, "clip_mae":clip_mae, "clip_rmse":clip_rmse, "video_r":video_r}

    bland_altman_plot(video_clip_gt_np, video_clip_pred_np, save_dir)
    return_dict = {"mae": video_mae, "rmse": video_rmse, "r": video_r, "step_log": step_log}
    return return_dict



class Loss(torch.nn.Module):
    def __init__(self, delta_t, K, fs):
        super(Loss, self).__init__()
        self.loss_func_p = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250,flag='p')
        self.loss_func_n = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250,flag='n')
        self.weight_dict = {"n_loss1": 1.0, "n_loss2": 1.0, "loss3": 1.0, "loss4": 1.0, "n_loss3": 1.0, "n_loss4": 1.0}

    def forward(self, model_output):
        model_output_1 = model_output[0:2]
        model_output_2 = model_output[2:]

    # positive/negative sample pairs for video 1
        model_arb_n_1 = torch.zeros_like(model_output_1)
        model_arb_n_1[0] = model_output_1[0]
        model_arb_n_1[1] = model_output_2[0]

        model_arb_n_3 = torch.zeros_like(model_output_1)
        model_arb_n_3[0] = model_output_1[0]
        model_arb_n_3[1] = model_output_2[1]

    # positive/negative sample pairs for video 2
        model_arb_n_2 = torch.zeros_like(model_output_2)
        model_arb_n_2[0] = model_output_1[1]
        model_arb_n_2[1] = model_output_2[1]

        model_arb_n_4 = torch.zeros_like(model_output_2)
        model_arb_n_4[0] = model_output_1[1]
        model_arb_n_4[1] = model_output_2[0]

        n_loss3 = self.loss_func_n(model_arb_n_3)
        n_loss4 = self.loss_func_n(model_arb_n_4)
        loss3 = self.loss_func_p(model_output_1)
        loss4 = self.loss_func_p(model_output_2)
        n_loss1 = self.loss_func_n(model_arb_n_1)
        n_loss2 = self.loss_func_n(model_arb_n_2)
        return {"n_loss1": n_loss1, "n_loss2": n_loss2, "loss3": loss3, "loss4": loss4, "n_loss3": n_loss3, "n_loss4": n_loss4}



def main(args):
    seed = int(args.seed * args.K)
    # init_seeds(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    exp_dir = args.exp_dir
    # logger = setup_logger(args.exp_dir)
    setup_logger(args.exp_dir, distributed_rank=get_rank(), filename="train_log.txt", mode="a")
    logger.info(f"Using seed {seed}")
    logger.info(f"Using {args.K} fold cross validation")
    logger.info(f"Saving to {exp_dir}")

    dataset = args.dataset
    if dataset == "pure":
        train_list, test_list = PURE_LU_split()
    elif dataset == "ubfc":
        train_list, test_list = UBFC_LU_split()
    elif dataset == "cohface":
        train_list, test_list = COHFACE_LU_split()
    if dataset == "cohface":
        fs = 20 # video frame rate, TODO: modify it if your video frame rate is not 30 fps.
    else:
        fs = 30
    T = fs * 10 # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2 # spatial dimenion of ST-rPPG block, default is 2x2.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T/2) # time length of each rPPG sample
    K = 4 # the number of rPPG samples at each spatial position

    dataset = H5Dataset(train_list, T,transform=transforms.Compose(
                [Normaliztion(), ToTensor()]))  # please read the code about H5Dataset when preparing your dataset


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,  # two videos for contrastive learning
                            shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the model and loss
    device_ids = [0,1]
    model = build_net(args, device).to(device).train()
    model = torch.nn.DataParallel(model,device_ids=device_ids)
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250,flag='normal')
    loss_func_p = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250,flag='p')
    loss_func_n = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250,flag='n')

    # define irrelevant power ratio
    # IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # assert False
    # j = 0
    history = {}
    start_epoch = 0
    for epoch in range(start_epoch, args.max_epochs+1):
        for it in range(np.round(60/(T/fs)).astype('int')): # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
            # random_scale_list = random.sample(scaleList, 2)
            # scale1 = random_scale_list[0]
            # scale2 = random_scale_list[1]
            #
            # # define the dataloader
            # dataset = H5Dataset(train_list, T)  # please read the code about H5Dataset when preparing your dataset
            # dataset.set_scale(scale1,scale2)
            # dataloader = DataLoader(dataset, batch_size=2,  # two videos for contrastive learning
            #                         shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            # metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            header = 'Epoch: [{}_{}]'.format(epoch, it)
            print_freq = 5
            logger.info("------------------------------------------------------!!!!")
            prefetcher = data_prefetcher(dataloader, device, prefetch=True)
            sample, targets = prefetcher.next()
            for _ in metric_logger.log_every(range(len(dataloader)), print_freq, header):
            # for sample in metric_logger.log_every(dataloader, print_freq, header):

            # for sample in dataloader: # dataloader randomly samples a video clip with length T
                imgs_scale1 = sample['img_seq_scale1']
                imgs_scale2 = sample['img_seq_scale2']
                scale1 = 128.0 / imgs_scale1.size(3)
                scale2 = 128.0 / imgs_scale2.size(3)
                # print(imgs_scale1.size())
                # print(imgs_scale2.size())

                # imgs_neg = torch.zeros((2,T,3,64,64))

                # for i in range(T):
                #     imgs_neg[0][i] = nearestInterp(imgs_lr[0][i])
                #     imgs_neg[1][i] = nearestInterp(imgs_lr[1][i])
                # imgs_neg = imgs_neg.permute((0, 2, 1, 3, 4))



                # inspect sample

                # array1 = imgs_scale1.permute((0, 2, 1, 3, 4))[:][0][0].cpu().numpy()
                # plt.imshow((array1.transpose((1, 2, 0))*128 + 127.5).astype('uint8'))
                # plt.show()
                # array1 = imgs_scale2.permute((0, 2, 1, 3, 4))[:][0][0].cpu().numpy()
                # plt.imshow((array1.transpose((1, 2, 0))*128 + 127.5).astype('uint8'))
                # plt.show()

                # default Gaussian blur parameters
                gass_kernel = 5
                sigma = 1.0

                if(args.gauss_blur):
                    h1 = imgs_scale1.size()[3]
                    h2 = imgs_scale2.size()[3]
                    if(h1 >= h2):
                        imgs_scale1 = imgs_scale1.permute((0, 2, 3, 4,1))* 128 + 127.5
                        for i in range(imgs_scale1.size()[0]):
                            for j in range(imgs_scale1.size()[1]):
                                array = imgs_scale1[i][j].cpu().numpy()
                                imgs_scale1[i][j] = torch.from_numpy(cv2.GaussianBlur(array, (gass_kernel, gass_kernel), sigma))
                        imgs_scale1 = imgs_scale1.permute((0, 4, 1, 2, 3))
                        imgs_scale1 = (imgs_scale1 - 127.5) / 128
                    else:
                        imgs_scale2 = imgs_scale2.permute((0, 2, 3, 4,1)) * 128 + 127.5
                        for i in range(imgs_scale2.size()[0]):
                            for j in range(imgs_scale2.size()[1]):
                                array = imgs_scale2[i][j].cpu().numpy()
                                imgs_scale2[i][j] = torch.from_numpy(cv2.GaussianBlur(array, (gass_kernel, gass_kernel), sigma))
                        imgs_scale2 = imgs_scale2.permute((0, 4, 1, 2, 3))
                        imgs_scale2 = (imgs_scale2 - 127.5) / 128

                imgs_scale1 = imgs_scale1.to(device) # B,T,C,H,W
                imgs_scale2 = imgs_scale2.to(device)
                # imgs_neg = imgs_neg.to(device)# B,T,C,H,W


                scale1 = torch.as_tensor(scale1).to(device)
                scale2 = torch.as_tensor(scale2).to(device)



                # model.set_scale(scale1,scale2)
                model.module.set_scale(scale1, scale2)
                # model forward propagation
                model_output = model(imgs_scale1,imgs_scale2)
                model_output_1 = model_output[0:2]
                model_output_2 = model_output[2:]

                # get rppg
                # rppg = model_output[:,-1]

                # # video1 positive sample pair
                # model_arb_p_1 = torch.zeros_like(model_output_1)
                # model_arb_p_1[0] = model_output_1[0]  # one upsampled 1, one upsampled 2
                # model_arb_p_1[1] = model_output_1[1]



                # # video2 positive sample pair
                # model_arb_p_2 = torch.zeros_like(model_output_2)
                # model_arb_p_2[0] = model_output_2[0]  # one upsampled 1, one upsampled 2
                # model_arb_p_2[1] = model_output_2[1]

                # positive/negative sample pairs for video 1
                model_arb_n_1 = torch.zeros_like(model_output_1)
                model_arb_n_1[0] = model_output_1[0]
                model_arb_n_1[1] = model_output_2[0]

                model_arb_n_3 = torch.zeros_like(model_output_1)
                model_arb_n_3[0] = model_output_1[0]
                model_arb_n_3[1] = model_output_2[1]

                # positive/negative sample pairs for video 2
                model_arb_n_2 = torch.zeros_like(model_output_2)
                model_arb_n_2[0] = model_output_1[1]
                model_arb_n_2[1] = model_output_2[1]

                model_arb_n_4 = torch.zeros_like(model_output_2)
                model_arb_n_4[0] = model_output_1[1]
                model_arb_n_4[1] = model_output_2[0]


                # #if you use one gpu
                # # get rppg
                # rppg_1 = model_output_1[:,-1]  # upsampled 1
                # rppg_2 = model_output_2[:, -1] # upsampled 2

                # # video1 positive sample pair
                # model_arb_p_1 = torch.zeros_like(model_output_1)
                # model_arb_p_1[0] = model_output_1[0]  # one upsampled 1, one upsampled 2
                # model_arb_p_1[1] = model_output_2[0]

                # # video2 positive sample pair
                # model_arb_p_2 = torch.zeros_like(model_output_2)
                # model_arb_p_2[0] = model_output_1[1]  # one upsampled 1, one upsampled 2
                # model_arb_p_2[1] = model_output_2[1]

                # # positive/negative sample pairs for video 1
                # model_arb_n_1 = torch.zeros_like(model_output_1)
                # model_arb_n_1[0] = model_output_1[0]
                # model_arb_n_1[1] = model_output_2[1]

                # # positive/negative sample pairs for video 2
                # model_arb_n_2 = torch.zeros_like(model_output_2)
                # model_arb_n_2[0] = model_output_1[1]
                # model_arb_n_2[1] = model_output_2[0]


                # define the loss functions
                n_loss3 = loss_func_n(model_arb_n_3)
                n_loss4 = loss_func_n(model_arb_n_4)
                loss3 = loss_func_p(model_output_1)
                loss4 = loss_func_p(model_output_2)
                n_loss1 = loss_func_n(model_arb_n_1)
                n_loss2 = loss_func_n(model_arb_n_2)


                loss = n_loss1 + n_loss2 + loss3 + loss4 + n_loss3 + n_loss4 #result1
                # loss = n_loss1  + loss3 + loss4 #result2
                # loss = n_loss2 + loss3 + loss4 #result3
                # loss = loss3 + loss4 + n_loss3 #result4
                # loss = loss3 + loss4 + n_loss4 #result5
                # loss = n_loss1 + n_loss2 + loss3 + loss4 #result6
                # loss = n_loss1 + n_loss2 + n_loss3 + loss3 + loss4 #result7

                # loss, p_loss, n_loss = loss_func(model_output)



                # array1 = imgs_hr_1.permute((0,1,4,2,3))[:][0][0].cpu().numpy()
                # array1 = array1.transpose((1, 2, 0)).astype('uint8')
                # plt.imshow(array1.transpose((1, 2, 0)).astype('uint8'))
                # plt.show()
                # plt.imsave("D:\pythonProject\实验\Arbitrary_Resolution_unsupervised_rPPG\sr_arb_1\ori.png",array1)


                # array1 = sr_arb1[:][0][0].cpu()
                # array1 = quantize(array1, 255).detach().numpy()
                # array1 = array1.transpose((0,1, 2)).astype('uint8')
                # # plt.imshow(array1.astype('uint8'))
                # # plt.show()
                # plt.imsave("D:\pythonProject\实验\Arbitrary_Resolution_unsupervised_rPPG\sr_arb_1\sr%d.png"%j, array1)
                # j += 1


                # loss_sr_ = torch.tensor(0.).to(device)
                # for i in range(T):
                #     ll = loss_sr(sr_arb1[0][i],imgs_hr_1[0][i])
                #     loss_sr_ = loss_sr_ + ll
                #     ll2 = loss_sr(sr_arb1[1][i],imgs_hr_1[1][i])
                #     loss_sr_ = loss_sr_ + ll2
                #
                # for i in range(T):
                #     ll = loss_sr(sr_arb2[0][i],imgs_hr_2[0][i])
                #     loss_sr_ = loss_sr_ + ll
                #     ll2 = loss_sr(sr_arb2[1][i],imgs_hr_2[1][i])
                #     loss_sr_ = loss_sr_ + ll2
                #
                # loss += loss_sr_ / 4 / T

                # optimize


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                sample, targets = prefetcher.next()

                # evaluate irrelevant power ratio during training
                # ipr_arb1 = torch.mean(IPR(rppg_1.clone().detach()))
                # ipr_arb2 = torch.mean(IPR(rppg_2.clone().detach()))
                # ipr = torch.mean(IPR(rppg.clone().detach()))



                # save loss values and IPR


            model.eval()
            save_dict = {"model": model.state_dict()}
            save_dict["optimizer"] = optimizer.state_dict()
            # save_dict["lr_scheduler"] = lr_scheduler.state_dict()
            save_dict["epoch"] = epoch
            # torch.save(save_dict, os.path.join(epcoh_dir, "latest.pt"))
            torch.save(save_dict, os.path.join(exp_dir, "latest.pt"))
            # val
            test_dict = evaluate(args, test_list, os.path.join(exp_dir, "latest.pt"), fs)
            # test_dict = evaluate_split_eq(args, dl_val, os.path.join(exp_dir, "latest.pt"), plot=True, save_dir=epcoh_dir)
            # # test_dict = evaluate_clip(args, dl_val, os.path.join(exp_dir, "latest.pt"), plot=True, save_dir=epcoh_dir, time_interval=44)
            # video_mae, video_rmse, video_r = test_dict["mae"], test_dict["rmse"], test_dict["r"]
            # lr_scheduler.step(video_mae)

            for name, metric in test_dict.items():
                history[name] = history.get(name, []) + [metric]
            logger.info(f"epoch: {epoch}, test_mae: {test_dict['mae']:.3f}, test_rmse: {test_dict['rmse']:.3f}, test_r: {test_dict['r']:.3f}")

            # best
            arr_scores = history["mae"]
            best_score_idx = np.argmin(arr_scores)
            if best_score_idx==len(arr_scores)-1:
                torch.save(save_dict, os.path.join(exp_dir, "best.pt"))
                logger.info("<<<<<< reach best {0} in epoch {1}: {2} >>>>>>".format(
                            "test_mae", best_score_idx, arr_scores[best_score_idx]))
            
            # Output the best epoch and its corresponding metrics
            best_mae = arr_scores[best_score_idx]
            best_rmse = history["rmse"][best_score_idx]
            best_r = history["r"][best_score_idx]
            logger.info(f"Current best epoch: {best_score_idx}, best_mae: {best_mae:.3f}, best_rmse: {best_rmse:.3f}, best_r: {best_r:.3f}")
            
            model.train()
            
if __name__ == "__main__":
    # %%
    parser_config = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser_config.add_argument('--K', type=int, default=1)
    parser_config.add_argument('--batch_size', type=int, default=2)
    parser_config.add_argument('--max_epochs', type=int, default=30)
    parser_config.add_argument('--model', type=str, required=True)
    parser_config.add_argument('--exp_name', type=str, required=False)
    parser_config.add_argument('--server_exp_dir', type=str, default="/home/hezhongtian/code/ARUR/result")
    # model
    parser_config.add_argument('--gauss_blur', type=str2bool, default=False)

    # dataset
    parser_config.add_argument('--dataset', type=str, default="pure")
    parser_config.add_argument('--T', type=int, default=160)
    parser_config.add_argument('--norm_type', type=str, default="normalize", choices=["normalize", "reconstruct"])
    parser_config.add_argument('--aug', type=str, default="f", help="figsc")
    parser_config.add_argument('--w', type=int, default=128)

    # loss
    parser_config.add_argument('--np_loss', type=str2bool, nargs='?', const=True, default=True)
    parser_config.add_argument('--np_weight', type=float, default=0.01)
    parser_config.add_argument('--cn_loss', type=str2bool, nargs='?', const=True, default=True)
    parser_config.add_argument('--cn_weight', type=float, default=1.0)
    parser_config.add_argument('--rec_loss', type=str2bool, nargs='?', const=True, default=False)

    # ---------------------  lr
    parser_config.add_argument('--lr', type=float, default=1e-4)
    parser_config.add_argument('--lr_patience', type=int, default=5)
    parser_config.add_argument('--lr_factor', type=float, default=0.5)
    parser_config.add_argument('--min_lr', type=float, default=1e-6)
    parser_config.add_argument('--weight_decay', type=float, default=5e-5)
    parser_config.add_argument('--seed', type=int, default=50)
    parser_config.add_argument('--print_i', type=int, default=1000)

    # ----------------------  resume
    parser_config.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False)
    parser_config.add_argument('--resume_path', type=str, default="")

    args = parser_config.parse_args()


    time_str = time.strftime("%m%d%H%M%S", time.localtime())
    args.exp_name = f"{args.model}_{time_str}_{args.exp_name}" # name_date_time
    args.exp_dir = os.path.join(args.server_exp_dir, f"{args.dataset.lower()}_{args.exp_name}", f"fold{args.K}_seed{args.seed}")


    if os.path.exists(args.exp_dir):
        print(f'rm -rf {os.path.join(args.server_exp_dir, f"{args.dataset.lower()}_{args.exp_name}")}')
        raise ValueError("exp_dir exists")
    else:
        os.makedirs(args.exp_dir, exist_ok=True)

    # backup code and command
    command_line_arguments = sys.argv
    terminal_command = " ".join(command_line_arguments[1:])
    with open(os.path.join(args.exp_dir, "terminal_command.sh"), "w") as f:
        f.write(f"python {__file__} {terminal_command}")

    cur_file = os.getcwd()
    cur_file_name = cur_file.split('/')[-1]
    os.makedirs(f'{args.exp_dir}/{cur_file_name}', exist_ok=True)
    for files in os.listdir(cur_file):
        if files not in ["__pycache__", "others", "using_file", "result"]:
            if os.path.isdir(f'{cur_file}/{files}'):
                shutil.copytree(f'{cur_file}/{files}', f'{args.exp_dir}/{cur_file_name}/{files}', dirs_exist_ok=True)
            else:
                shutil.copy(f'{cur_file}/{files}', f'{args.exp_dir}/{cur_file_name}/{files}')

    # tmux name
    import subprocess
    # run tmux command to get current session name
    try:
        result = subprocess.run(['tmux', 'display-message', '-p', '#S'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        tmux_session = result.stdout.strip()
    # print(f"Current tmux session name for this terminal: {tmux_session}----------------{result.stdout}")
        with open(os.path.join(args.exp_dir, f"tmux_{tmux_session}.txt"), "w") as f:
            f.write(f"python {__file__} {terminal_command}")
    except subprocess.CalledProcessError:
        print("Not in a tmux session")

    main(args)

# CUDA_VISIBLE_DEVICES=0,1 python train.py --model PhysNet_sda --exp_name 1919 --dataset
# https://blog.csdn.net/qq_37289115/article/details/122742608logger.info(f"epoch: {epoch}, test_mae: {test_dict['mae']:.3f}, test_rmse: {test_dict['rmse']:.3f}, test_r: {test_dict['r']:.3f}")