import cv2
import os
import json
import math
import copy
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import h5py
from scipy import interpolate
from tqdm import tqdm
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
import concurrent
import sys

sys.path.append("../../")
from utils.signal import cal_hr
# from accelerate.logging import get_logger
from utils.logger import get_logger
import datasets.transforms as transforms
import random

logger = get_logger(__name__)


# Return T frames or the whole video.
# Each dataset implements its own video/frame reading and frame-rate handling.
# T = -1 means load the entire video.
class BaseDataset(Dataset):
    def __init__(self, data_dir, train=True, T=-1, transform_rate=30, transforms_list=None, w=64, h=64,
                 limit_sample_num=0, notAll=True, data_name=None, video_random_num=0, \
                 norm_type="reconstruct", cache_dir="/root/tmp_rppg/tmp3", use_seg=False, use_shading=False, kfold=1,
                 aug=""):
        """
    :param data_dir: dataset root directory
    :param train: whether this is a training split
    :param T: number of frames to read; -1 means read the whole video
    :param transform_rate: target frame rate for returned video and ECG sampling rate
        """
        # init params
        self.data_dir = data_dir  # data_dir: dataset root directory
        self.name = data_name  # data_name: dataset identifier
        self.train = train  # train: True for training split, False for val/test
        self.T = T  # T: number of frames to read; -1 means read the whole video
        self.transform_rate = transform_rate  # transform_rate: returned video frame rate and ECG sampling rate
        self.norm_type = norm_type
        self.transforms = 1  # transforms_list: preprocessing / augmentation transforms
        #transforms.Compose([Normaliztion(self.norm_type)])  #transforms_list
        self.w = w  # w: video frame width (pixels)
        self.h = h  # h: video frame height (pixels)
        self.kfold = kfold  # kfold: folds used for K-fold cross validation
        # norm_type: normalization mode, default is "reconstruct"
        if self.norm_type == "reconstruct":
            assert self.w == 64 and self.h == 64, "For 'reconstruct' mode, w and h must be 64"
        else:
            assert self.w == 64 and self.h == 64, "For 'normalize' mode, w and h must be 128"
        self.limit_sample_num = limit_sample_num  # limit_sample_num: limit number of loaded samples
        self.rate_threshold = 3  # rate_threshold: resample when frame rate difference exceeds this
        self.video_random_num = video_random_num  # video_random_num: control randomness when loading videos
        self.data_list = list()
        self.notAll = notAll  # notAll: whether to load all data (False loads only a subset)
        self.use_seg = use_seg  # use_seg: whether segmentation maps are used
        self.use_shading = use_shading  # use_shading: whether shading maps are used
        self.aug = aug  # aug: augmentation flags string

        # populate data_list: returns a list where each element is a dict containing
        # frame_path (list), ecg, frame_rate (original), ecg_rate (original)
        self.get_data_list()

        # backup datalist and split by limit_sample_num
        self.backup_data_list = copy.deepcopy(self.data_list)
        if self.limit_sample_num > 0:
            original_len = len(self.backup_data_list)
            select_index = np.random.choice(original_len, self.limit_sample_num, replace=False)
            self.data_list = [self.backup_data_list[i] for i in select_index]
        # self.data_list = self.backup_data_list + self.backup_data_list[:int(0.5*len(self.backup_data_list))]
        # add cache h5
        self.cache_dir = cache_dir  # cache_dir: directory to store cache files
        # os.makedirs(self.cache_dir, exist_ok=True)
        if self.data_dir[-1] == "/":
            use_data_dir = self.data_dir[:-1]
        else:
            use_data_dir = self.data_dir
        self.cache_name = f"{self.cache_dir}/{self.name}{use_data_dir.replace('/', '_')}_cache.h5"  # <dataset>_<path>_cache.h5
        # check whether cache file exists
        self.can_use_cache = False
        self.speed_slow = 0.6 # slow speed coefficient
        self.speed_fast = 1.4 # fast speed coefficient
        self.set_augmentations()
        # if not os.path.exists(self.cache_name):
        #     # print(f"cache file {self.cache_name} not exists, please create it first")
        #     self.can_use_cache = False
        # else:
        #     self.can_use_cache = True
        #     # print(f"cache file {self.cache_name} exists, use it")

    def create_cache(self):
        """
        Create cache when T == -1.
        Save the processed video and ECG data into an HDF5 cache file. This
        caching can significantly speed up subsequent data loading, especially
        for large-scale video datasets.
        """

        backup_T = self.T
        backup_notAll = self.notAll
        self.notAll = False
        self.T = -1
        self.transforms = None
        self.can_use_cache = False
        self.data_list = []
        self.get_data_list()
        # create cache
        """
        Purpose: open an HDF5 file, iterate over the data list and process each
        sample one by one.

        Details:
        - __getitem__(i): fetch video, ECG and related fields by index.
        - Extract video and ecg from the sample.
        - Read frame_rate and ecg_rate metadata.
        - If use_seg or use_shading are enabled, also extract segmentation and
          shading video data.
        """
        with h5py.File(self.cache_name, "w") as f:
            for i, data in enumerate(tqdm(self.data_list)):
                name = "_".join(data['location'].split('_')[:-2])
                sample = self.__getitem__(i)
                video = sample['video']  # c, t, h, w
                ecg = sample['ecg']
                if self.use_seg:
                    seg_video = sample['seg']  # t, h, w
                if self.use_shading:
                    shading_video = sample['shading']  # t, h, w
                frame_rate = data["frame_rate"]
                ecg_rate = data["ecg_rate"]

                """
                Purpose: if the video's frame rate differs from the target
                frame rate by more than `rate_threshold`, resample the video.

                Process: use PyTorch's interpolate to change the temporal
                dimension so that the number of frames matches
                `transform_rate`. Convert resampled data to numpy for storage.
                """
                if abs(self.transform_rate - frame_rate) > self.rate_threshold:
                    # video = torch.from_numpy(video)
                    original_len = video.shape[1]
                    video = torch.nn.functional.interpolate(video.unsqueeze(0), size=(
                        math.ceil(original_len * self.transform_rate / frame_rate), self.h, self.w),
                                                            mode="trilinear").squeeze(0)
                    video = video.numpy()  # c, t, h, w

                """
                Purpose: make sure segmentation (seg_video), shading_video and
                ECG have the same temporal length as the video.

                Process: resample seg_video and shading_video accordingly, and
                linearly interpolate the ECG to match the video frame count.
                """
                # adjust ecg, seg, shading according to video
                if self.use_seg and seg_video.shape[0] != video.shape[1]:
                    # seg_video = torch.from_numpy(seg_video)
                    seg_video = torch.nn.functional.interpolate(seg_video.unsqueeze(0).unsqueeze(0),
                                                                size=(video.shape[1], self.h, self.w),
                                                                mode="trilinear").reshape(-1, self.h, self.w)
                    seg_video = seg_video.numpy()
                if self.use_shading and shading_video.shape[0] != video.shape[1]:
                    # shading_video = torch.from_numpy(shading_video)
                    shading_video = torch.nn.functional.interpolate(shading_video.unsqueeze(0).unsqueeze(0),
                                                                    size=(video.shape[1], self.h, self.w),
                                                                    mode="trilinear").reshape(-1, self.h, self.w)
                    shading_video = shading_video.numpy()
                if ecg.shape[0] != video.shape[1]:
                    # ecg = torch.from_numpy(ecg.astype(np.float32))
                    ecg = torch.nn.functional.interpolate(ecg.view(1, 1, -1), size=(video.shape[1],),
                                                          mode="linear").reshape(-1)
                    ecg = ecg.numpy()

                """
                Purpose: convert processed video, ECG and other arrays into a
                storage-friendly format and write them to the HDF5 cache.

                Process: reorder video to (t, h, w, c) and cast to uint8; cast
                ECG to float16. If present, cast seg/shading to uint8. Create
                corresponding HDF5 datasets and save metadata such as frame
                and ECG sampling rates.
                """
                video = video.transpose(1, 2, 3, 0).astype(np.uint8)  # t, h, w, c
                ecg = ecg.astype(np.float16)  # t
                if self.use_seg:
                    seg_video = seg_video.astype(np.uint8)
                if self.use_shading:
                    shading_video = shading_video.astype(np.uint8)
                # add h5file
                gp = f.create_group(name)
                gp.create_dataset("video", data=video, chunks=(10, self.h, self.w, 3))
                gp.create_dataset("ecg", data=ecg)
                if self.use_seg:
                    gp.create_dataset("seg", data=seg_video, chunks=(10, self.h, self.w))
                if self.use_shading:
                    gp.create_dataset("shading", data=shading_video, chunks=(10, self.h, self.w))
                gp.attrs["frame_rate"] = frame_rate
                gp.attrs["ecg_rate"] = ecg_rate
        # Restore original T and notAll parameters so subsequent operations are unaffected.
        self.T = backup_T
        self.notAll = backup_notAll

    def get_data_list(self):
        raise NotImplementedError

    def __str__(self):
        return self.name  # Return the dataset name

    def __len__(self):
        return len(self.data_list)

    def open_hdf5(self):
        self.h5f = h5py.File(self.cache_name, 'r')
        # self.h5f_shading = h5py.File(self.shading_file, 'r')
        # self.dataset = self.img_hdf5['dataset'] # if you want dataset.

    def set_augmentations(self):
        """
        Configure augmentation flags according to self.aug string.
        Supported flags:
        'f' - horizontal flip
        'i' - illumination perturbation
        'g' - gaussian noise
        's' - speed/time resampling
        'c' - random resized crop
        """
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        self.aug_speed = False
        self.aug_resizedcrop = False
        if self.train:
            self.aug_flip = True if 'f' in self.aug else False
            self.aug_illum = True if 'i' in self.aug else False
            self.aug_gauss = True if 'g' in self.aug else False
            self.aug_speed = True if 's' in self.aug else False
            self.aug_resizedcrop = True if 'c' in self.aug else False
        self.aug_reverse = False  ## Don't use this with supervised
        # self.aug_flip = True
        # self.aug_illum = True
        # self.aug_gauss = True
        # self.aug_speed = True
        # self.aug_resizedcrop = True
        # self.aug_reverse = True

    def get_sample(self, index):
        """
        Fetch video, ECG and related fields from dataset, apply transforms and
        adjust lengths to match the requested number of frames. Returns a dict:
        video: (c, t, h, w),
        ecg: (t,),
        clipAverageHR: scalar,
        seg: (t, h, w) in [0,1] if present,
        shading: (t, h, w) in [0,1] if present,
        location: identifier string
        """
        if self.can_use_cache:
            # if not hasattr(self, 'h5f'):
            #     self.open_hdf5()
            name = "_".join(self.data_list[index]['location'].split('_')[:-2])
            start_idx = int(self.data_list[index]['location'].split('_')[-2])
            end_idx = int(self.data_list[index]['location'].split('_')[-1])
            location = f"{name}_{start_idx}_{end_idx}"

            # Compute resampled indices based on clip_idx
            clip_idx = self.data_list[index]['clip_idx']
            # Purpose: Load video, ECG, segmentation and shading data from the
            # cache file using the provided index and frame information.
            #
            # Process:
            # - If T != -1: compute the frame range using the clip index and
            #   load the specified frames.
            # - If T == -1: load the entire video and associated signals.
            # - After loading, retrieve stored metadata such as frame rate and
            #   ECG sampling rate.
            with h5py.File(self.cache_name, 'r') as h5f:
                if self.T != -1:
                    start_idx = clip_idx * self.T
                    end_idx = (clip_idx + 1) * self.T
                    video = np.array(h5f[name]['video'][start_idx: end_idx]).astype("float32")  # t, h, w, c
                    ecg = np.array(h5f[name]['ecg'][start_idx: end_idx]).astype("float32")
                    if self.use_seg:
                        seg_video = np.array(h5f[name]['seg'][start_idx: end_idx]).astype("float32")
                    if self.use_shading:
                        shading_video = np.array(h5f[name]['shading'][start_idx: end_idx]).astype("float32")
                else:
                    video = np.array(h5f[name]['video']).astype("float32")
                    ecg = np.array(h5f[name]['ecg']).astype("float32")
                    if self.use_seg:
                        seg_video = np.array(h5f[name]['seg']).astype("float32")
                    if self.use_shading:
                        shading_video = np.array(h5f[name]['shading']).astype("float32")
                frame_rate = h5f[name].attrs["frame_rate"]
                ecg_rate = h5f[name].attrs["ecg_rate"]
        else:
            # Purpose: If cache is not used, load video, ECG, segmentation and
            # shading data directly from the original file paths.
            #
            # Process:
            # - Use read_video and read_video_2d helpers to read images from
            #   disk and return arrays.
            # - After loading, retrieve frame rate and ECG sampling rate
            #   information from the dataset entry.
            video = read_video(self.data_list[index]["frame_path"]).astype("float32")  # t, h, w, c
            ecg = self.data_list[index]["ecg"].astype("float32")
            if self.use_seg:
                seg_video = read_video_2d(self.data_list[index]["seg_path"]).astype("float32")
            if self.use_shading:
                shading_video = read_video_2d(self.data_list[index]["shading_path"]).astype("float32")
            frame_rate = self.data_list[index]["frame_rate"]
            ecg_rate = self.data_list[index]["ecg_rate"]
            location = self.data_list[index]["location"]
        video = torch.from_numpy(video).permute(3, 0, 1, 2)  # c, t, h, w
        ecg = torch.from_numpy(ecg)  # t,
        if self.use_seg:
            seg_video = torch.from_numpy(seg_video)
        if self.use_shading:
            shading_video = torch.from_numpy(shading_video)

        """
        Normalize or preprocess video, ECG, segmentation and shading arrays.
        - 'normalize': scale video pixels to [-1, 1]
        - 'reconstruct': scale to [0,1] and apply gamma correction
        ECG is zero-centered and scaled by max absolute value. Segmentation/shading
        are scaled to [0,1].
        """
        if self.transforms is not None:
            if self.norm_type == "normalize":
                video = video.add_(-127.5).div_(127.5)
            if self.norm_type == "reconstruct":
                video = video.div_(255.0).pow_(2.4)
            if self.name != "MMSE-HR":
                ecg = (ecg - ecg.mean()) / ecg.abs_().max()
            if self.use_seg:
                seg_video = seg_video.div_(255.0)
            if self.use_shading:
                shading_video = shading_video.div_(255.0)
                """
                Purpose: Adjust video, ECG, segmentation and shading arrays to match
                the requested number of frames (T) or the target frame rate
                (transform_rate).

                Process:
                - If T != -1: resample or crop to exactly T frames.
                - If T == -1: resample according to transform_rate to determine the
                    new frame count.
                """
        if not self.can_use_cache:
            if self.T != -1:
                # Resampling to T frames
                if video.shape[1] != self.T:
                    video = torch.nn.functional.interpolate(video.unsqueeze(0), size=(self.T, self.h, self.w),
                                                            mode="trilinear").squeeze(0)
                if ecg.shape[0] != self.T:
                    ecg = torch.nn.functional.interpolate(ecg.view(1, 1, -1), size=(video.shape[1],),
                                                          mode="linear").reshape(-1)
                if self.use_seg and seg_video.shape[0] != self.T:
                    seg_video = torch.nn.functional.interpolate(seg_video.unsqueeze(0).unsqueeze(0),
                                                                size=(self.T, self.h, self.w),
                                                                mode="trilinear").reshape(-1, self.h, self.w)
                if self.use_shading and shading_video.shape[0] != self.T:
                    shading_video = torch.nn.functional.interpolate(shading_video.unsqueeze(0).unsqueeze(0),
                                                                    size=(self.T, self.h, self.w),
                                                                    mode="trilinear").reshape(-1, self.h, self.w)
            else:
                # Resampling according to target frame rate
                new_T = math.ceil(video.shape[1] * self.transform_rate / frame_rate)
                if abs(frame_rate - self.transform_rate) > self.rate_threshold:
                    video = torch.nn.functional.interpolate(video.unsqueeze(0), size=(new_T, self.h, self.w),
                                                            mode="trilinear").squeeze(0)
                if ecg.shape[0] != video.shape[1]:
                    ecg = torch.nn.functional.interpolate(ecg.view(1, 1, -1), size=(video.shape[1],),
                                                          mode="linear").reshape(-1)
                if self.use_seg and seg_video.shape[0] != video.shape[1]:
                    seg_video = torch.nn.functional.interpolate(seg_video.unsqueeze(0).unsqueeze(0),
                                                                size=(new_T, self.h, self.w), mode="trilinear").reshape(
                        -1, self.h, self.w)
                if self.use_shading and shading_video.shape[0] != video.shape[1]:
                    shading_video = torch.nn.functional.interpolate(shading_video.unsqueeze(0).unsqueeze(0),
                                                                    size=(new_T, self.h, self.w),
                                                                    mode="trilinear").reshape(-1, self.h, self.w)

        # calculate clipAverageHR
        # Purpose: Compute the average heart rate for the current sample and
        # organize the processed data into a dictionary for return.
        #
        # Process:
        # - Call cal_hr to compute average heart rate from the ECG signal.
        # - Pack video, ECG, heart rate, location and sampling rates into a
        #   dictionary named `sample`.
        # - If segmentation or shading data exist, include them in the
        #   dictionary as well.
        #
        # Returns:
        # - sample (dict): processed data payload for the sample.
        clipAverageHR, _, _ = cal_hr(ecg.reshape(1, -1))
        # to one sample dict
        sample = {"video": video, "ecg": ecg, "clipAverageHR": clipAverageHR, "location": location,
                  "frame_rate": frame_rate, "ecg_rate": ecg_rate, "frameRate": 30}
        if self.use_seg:
            sample["seg"] = seg_video
        if self.use_shading:
            sample["shading"] = shading_video
        return sample

    def valid_data(self, data):
        """
        Validate a sample's ECG and heart rate values. Return False if the
        sample is invalid.

        Checks performed:
        - ECG contains NaNs -> invalid
        - clipAverageHR outside reasonable bounds (40-180 bpm) -> invalid
        """
        if np.isnan(data["ecg"]).any():
            logger.info(f"ecg has NaN values: {data['location']}")
            return False
        if data["clipAverageHR"] > 180 or data["clipAverageHR"] < 40:
            logger.info(f"clipAverageHR out of range: {data['location']}")
            return False
        return True

    def __getitem__(self, index):
        sample = self.get_sample(index)
    # if self.train:  # Only needed during training
        #     while not self.valid_data(sample):
        #         self.del_indexlist.append(index)
        #         tmp_list = list(set(self.real_indexlist) - set(self.del_indexlist))
        #         select_index = random.choice(tmp_list)
        #         sample = self.get_sample(select_index)
        return sample

    """
    Read a segment of frames from a video starting at the specified frame
    index. For each frame, perform face detection, crop the face region and
    resize to the configured output size. Returns the processed frame array.
    """
    def read_video_from_video(self,
                              frame_path):  #/data/public_dataset/BUAA-MIHR/Sub 01/lux 100.0/lux100.0_APH.avi/941.png
        """
    Load a pretrained face detector (haarcascade_frontalface_default.xml)
    using OpenCV's CascadeClassifier. Disable OpenCL and multi-threading
    for deterministic behavior.
        """
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        video_x = np.zeros((len(frame_path), self.w, self.h, 3))
    # Open video file
        start_frame = int(frame_path[0].split("/")[-1].split(".png")[0])

        video_path = frame_path[0].split(".avi")[0] + ".avi"
        cap = cv2.VideoCapture(video_path)
        f = True
        x, y, w, h = -1, 182, 151, 151
    # Number of frames to skip before starting
        for i in range(start_frame):
            _, _ = cap.read()

    # Read frames from x to x+160 and store them into the numpy buffer
        for i in range(self.T):
            ret, frame = cap.read()
            if ret:
                tmp_image = frame
                if (f == True):
                    gray = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if (faces != ()):
                        for (_x, _y, _w, _h) in faces:
                            if (_x != 0):
                                x, y, w, h = _x, _y, _w, _h
                                # print("s",i)
                        f = False
                    if (x == -1):
                        tmp_image = tmp_image
                    else:
                        tmp_image = tmp_image[y:y + h, x:x + w]
                imageRGB = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
            else:
                break
            video_x[i, :, :, :] = cv2.resize(imageRGB, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        return video_x

    # fps = frame_rate = frame_num / duration, duration = frame_num / frame_rate, frame_num = duration * frame_rate
    # The duration before and after resampling is the same, therefore:
    # resampled_length = original_length * (resampled_rate / original_rate)
    # Given a target frame count T after resampling, the original length is:
    # original_length = T * (original_rate / resampled_rate)
    def calculate_chunk_split(self, old_length, old_rate, new_rate, T):
        """
        Split a video into segments that meet the specified frame count T.
        If the target frame rate differs from the original, resampling is
        applied to compute appropriate segment boundaries.

        Parameters:
        - old_length: number of frames in the original video.
        - old_rate: original frame rate.
        - new_rate: target frame rate.
        - T: segment length in frames (time steps per segment). If T == -1,
             the whole video is returned as a single segment.
        """
        if T == -1:
            return [[0, old_length]]

        # If the original frame rate is close enough to the target rate,
        # do not resample and simply split by T frames.
        if abs(self.transform_rate - old_rate) <= self.rate_threshold:
            chunk_split = list(range(0, old_length - T + 1, T))
            chunk_list = []
            for chunk in chunk_split:
                if chunk + T <= old_length:
                    chunk_list.append([chunk, chunk + T])
            return chunk_list

        # Otherwise, compute splits with resampling taken into account.
        new_length = round(old_length * new_rate / old_rate)  # Resampled video length (rounded)
        if new_length < T:
            raise ValueError("video length is too short")
        else:
            # Map resampled frames back to original frames (ceil-like) to avoid
            # edge cases and to match temporal interpolation math.
            map_to_old = int(T * old_rate / new_rate) + 1
            chunk_split = list(range(0, old_length - map_to_old + 1, map_to_old))
            chunk_list = []
            for chunk in chunk_split:
                if chunk + map_to_old <= old_length:
                    chunk_list.append([chunk, chunk + map_to_old])
            return chunk_list


"""
Read and process a list of image files: convert to RGB images of the target size and
store them in a numpy array for return.
"""
def read_video(frame_path):
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(0)

    video_x = np.zeros((len(frame_path), 64, 64, 3), dtype=np.uint8)

    # def fetch_image(image_index, image_path, num_retries: int = 10):

    #     imageBGR = cv2.imread(image_path)

    #     try:
    #         imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    #     except:
    #         assert False, f"frame path: {image_path} is wrong"
    #     if imageRGB.shape[0] != self.w or imageRGB.shape[1] != self.h:
    #         imageRGB = cv2.resize(imageRGB, (self.w, self.h))
    #     video_x[image_index] = imageRGB

    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     task_list = [executor.submit(fetch_image, i, frame) for i, frame  in enumerate(frame_path)]

    #     for item in concurrent.futures.as_completed(task_list):
    #         item.result()

    for i, frame in enumerate(frame_path):
        imageBGR = cv2.imread(frame)
        try:
            imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        except:
            assert False, f"frame path: {frame} is wrong"
        if imageRGB.shape[0] != 64 or imageRGB.shape[1] != 64:
            imageRGB = cv2.resize(imageRGB, (64, 64), interpolation=cv2.INTER_CUBIC)
        video_x[i, :, :, :] = imageRGB
    # print(id(video_x))
    # print(video_x.data)
    return video_x

"""
Read and process a list of 2D images (segmentation/mask). Load as grayscale
and resize to the target size (64x64). Returns a numpy array containing the
loaded masks.
"""
def read_video_2d(seg_path):
    # Read segmentation/mask or other 2D mask images
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(0)
    video_x = np.zeros((len(seg_path), 64, 64))
    # def fetch_image(image_index, image_path, num_retries: int = 10):
    #     try:
    #         imageGray = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #         if imageGray.shape[0] != self.w or imageGray.shape[1] != self.h:
    #             imageGray = cv2.resize(imageGray, (self.w, self.h))
    #     except:
    #         assert False, f"frame path: {image_path} is wrong"

    #     video_x[image_index] = imageGray

    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     task_list = [executor.submit(fetch_image, i, frame) for i, frame  in enumerate(seg_path)]

    #     for item in concurrent.futures.as_completed(task_list):
    #         item.result()
    for i, frame in enumerate(seg_path):
        imageGray = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        if imageGray.shape[0] != 64 or imageGray.shape[1] != 64:
            imageGray = cv2.resize(imageGray, (64, 64), interpolation=cv2.INTER_CUBIC)
        video_x[i, :, :] = imageGray
    return video_x


class PURE(BaseDataset):

    def get_data_list(self):
        self.name = "PURE"
        date_list = os.listdir(self.data_dir)
        try:
            date_list.remove('pure')
        except:
            pass
        date_list.sort()
        if self.notAll:
            if self.train:
                date_list = [subject for subject in date_list if subject[:2] not in ["02", "03", "10"]]
            else:
                date_list = [subject for subject in date_list if subject[:2] in ["02", "03", "10"]]
        # print("date_list: ", date_list)
        for date in date_list:
            video_dir = os.path.join(self.data_dir, date)

            # read label
            json_file = os.path.join(self.data_dir, date, date + ".json")
            with open(json_file, 'r') as f:
                data = json.load(f)
            ecg_time_stamp = np.array([i['Timestamp'] for i in data['/FullPackage']])
            ecg = np.array([i['Value']['waveform'] for i in data['/FullPackage']])
            video_time_stamp = np.array([i['Timestamp'] for i in data['/Image']])
            # Ensure frame file order is correct using timestamps from the JSON file
            # Also collect segmentation and shading paths
            frame_path = []
            seg_path = []
            shading_path = []
            for i in range(len(video_time_stamp)):
                frame_path.append(os.path.join(video_dir, date, f"Image{video_time_stamp[i]}.png"))
                seg_path.append(os.path.join(video_dir, 'seg', f"Image{video_time_stamp[i]}.jpg"))
                shading_path.append(os.path.join(video_dir, 'shading', f"Image{video_time_stamp[i]}.jpg"))

            assert len(ecg_time_stamp) == len(ecg)

            ecg_time_diffs = np.diff(ecg_time_stamp / 1e9)
            ecg_rate = 1 / ecg_time_diffs.mean()
            frame_time_diffs = np.diff(video_time_stamp / 1e9)
            frame_rate = 1 / frame_time_diffs.mean()

            ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, self.transform_rate, self.T)
            frame_chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)

            # assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"
            for idx in range(len(ecg_chunk_split)):
                start_idx, end_idx = ecg_chunk_split[idx]
                ecg_select = ecg[start_idx:end_idx]
                start_idx, end_idx = frame_chunk_split[idx]
                # print(idx, start_idx, end_idx)
                frame_select = frame_path[start_idx:end_idx]
                self.data_list.append(
                    {'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': ecg_rate, \
                     "location": f"{date}_{start_idx}_{end_idx}", "clip_idx": idx, \
                     'seg_path': seg_path[start_idx:end_idx], 'shading_path': shading_path[start_idx:end_idx], \
                     })


class UBFC(BaseDataset):
    def get_data_list(self):
        self.name = "UBFC"
        subject_list = os.listdir(self.data_dir)
        subject_list.remove('subject11')
        subject_list.remove('subject18')
        subject_list.remove('subject20')
        subject_list.remove('subject24')  # This video causes negative heart rate values in PhysNet (after subtracting 40)
        subject_list.sort()
        if self.notAll:
            if self.train:
                subject_list = ["subject" + str(i) for i in range(1, 42)]
                subject_list.remove('subject20')
                subject_list.remove('subject11')
                subject_list.remove('subject18')
                subject_list.remove('subject24')
                subject_list.remove('subject2')
                subject_list.remove('subject6')
                subject_list.remove('subject7')
                subject_list.remove('subject19')
                subject_list.remove('subject21')

                subject_list.remove('subject28')
                subject_list.remove('subject29')
            else:
                subject_list = ["subject" + str(i) for i in range(42, 50)]
        for subject in subject_list:
            video_dir = os.path.join(self.data_dir, subject)
            frame_list = os.listdir(video_dir + '/pic/')
            # frame_list.remove("ground_truth.txt")
            frame_list_int = [int(i.split(".")[0]) for i in frame_list]
            frame_list_int.sort()  # Convert to integers to ensure correct ordering of frame filenames
            frame_path = []
            seg_path = []
            shading_path = []
            for frame_name in frame_list_int:
                frame_path.append(os.path.join(video_dir, 'pic', f"{frame_name:05d}.png"))
                seg_path.append(os.path.join(video_dir, 'seg', f"{frame_name:05d}.png"))
                shading_path.append(os.path.join(video_dir, 'shading', f"{frame_name:05d}.png"))
            # read label
            with open(os.path.join(video_dir, 'ground_truth.txt'), 'r') as f:
                data = f.readlines()
            data_timestamp = np.array(
                [float(strr.replace('e', 'E')) for strr in list(data[2].split())])  # Per-frame timestamps in seconds; start timestamp is 0
            data_Hr = np.array([float(strr.replace('e', 'E')) for strr in list(data[1].split())])
            data_ecg = np.array([float(strr.replace('e', 'E')) for strr in list(data[0].split())])
            assert len(data_timestamp) == len(data_Hr) == len(data_ecg)
            assert len(data_timestamp) == len(frame_path)
            time_diffs = np.diff(data_timestamp)
            frame_rate = 1 / time_diffs.mean()
            chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)

            for clip_idx, (start_idx, end_idx) in enumerate(chunk_split):
                frame_select = frame_path[start_idx:end_idx]
                ecg_select = data_ecg[start_idx:end_idx]
                seg_select = seg_path[start_idx:end_idx]
                shading_select = shading_path[start_idx:end_idx]
                self.data_list.append(
                    {'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': frame_rate, \
                     "location": f"{subject}_{start_idx}_{end_idx}", "clip_idx": clip_idx, \
                     'seg_path': seg_select, 'shading_path': shading_select})


class V4V(BaseDataset):
    def __init__(self, data_dir, train=True, T=-1, transform_rate=30, limit_batches=0, notAll=True):
        super().__init__(data_dir, train, T, transform_rate, limit_batches=limit_batches, notAll=notAll)

    def get_data_list(self):
        path = "/data/maoguanhui/Phase1_Training_Validationsets/"
        videoPath = path + "/Videos/"
        videoList001 = [videoPath + "/train_001of002/" + v + '/pic/'
                        for v in os.listdir(videoPath + "/train_001of002/")]
        videoList002 = [videoPath + "/train_002of002/" + v + '/pic/'
                        for v in os.listdir(videoPath + "/train_002of002/")]
        HRPath = path + "Ground truth/Physiology/"
        listTrain = os.listdir(HRPath)
        PhysPathList = [HRPath + item for item in listTrain]

        subject_list = videoList001 + videoList002
        if self.notAll:
            if self.train:
                subject_list = videoList001

            else:
                subject_list = videoList002
        for subject in subject_list:
            #video_dir = os.path.join(self.data_dir, subject)
            frame_list = os.listdir(subject)
            # frame_list.remove("ground_truth.txt")
            frame_list_int = [int(i.split(".")[0]) for i in frame_list]
            if len(frame_list_int) < self.T:
                continue

            frame_list_int.sort()  # Convert to integers to ensure correct ordering of frame filenames
            frame_path = []
            for frame_name in frame_list_int:
                frame_path.append(os.path.join(subject, f"{frame_name}.png"))
            # read label
            target_path = "/data/maoguanhui/Phase1_Training_Validationsets/Ground truth/Physiology/" + \
                          subject.split('/')[-3] + '.txt'

            # ind = PhysPathList.index(target_path)
            with open(target_path, "r") as f:
                HR_temp = f.readlines()[0]
                HR_list = HR_temp.split(",")
                vName = HR_list.pop(0)
                kind = HR_list.pop(0)
                HR_list[-1] = HR_list[-1].strip("\n")
                dataHr = [float(i) for i in HR_list]
            data_ecg = np.array(dataHr)

            # print(target_path)
            # print(len(frame_path),frame_path[0])
            # print(len(dataHr),vName,kind)

            # /data/maoguanhui/Phase1_Training_Validationsets/Ground truth/Physiology/F052_T6.txt
            # 1803 /data/maoguanhui/Phase1_Training_Validationsets//Videos//train_001of002/F052_T6/pic/0.png
            # 1803 F052_T6.mkv  HR
            frame_rate = 25
            chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
            for start_idx, end_idx in chunk_split:
                frame_select = frame_path[start_idx:end_idx]
                ecg_select = data_ecg[start_idx:end_idx]
                self.data_list.append(
                    {'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': frame_rate})


class COHFACE(BaseDataset):
    def __init__(self, data_dir, train=True, T=-1, transform_rate=30, limit_batches=0, notAll=True):
        super().__init__(data_dir, train, T, transform_rate, limit_batches=limit_batches, notAll=notAll)

    def get_data_list(self):
        date_list = os.listdir(self.data_dir)
        date_list.remove("protocols")
        date_list.remove(".README.rst.swp")
        date_list.remove("README.rst")
        date_list.sort()
        if self.notAll:
            if self.train:
                date_list = [subject for subject in date_list if
                             int(subject) not in [25, 29, 31, 35, 27, 33, 1, 16, 38, 21, 28, 4]]
            else:
                date_list = [str(subject) for subject in [25, 29, 31, 35, 27, 33, 1, 16, 38, 21, 28, 4]]
        # print("date_list: ", date_list)
        for i in range(len(date_list)):
            #videoCap = cv2.VideoCapture(self.path + self.videoList[i] + '/001vid.avi')
            for vd in range(4):
                video_dir = self.data_dir + date_list[i] + '/' + str(vd)
                frame_list = os.listdir(video_dir + '/pic/')
                # frame_list.remove("ground_truth.txt")
                frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                frame_list_int.sort()  # Convert to integers to ensure correct ordering of frame filenames
                frame_path = []
                for frame_name in frame_list_int:
                    frame_path.append(os.path.join(video_dir, 'pic', f"{frame_name}.png"))
                # read label
                with h5py.File(video_dir + "/data.hdf5", 'r') as f:
                    data_ecg = f['pulse'][:]
                    data_timestamp = f['time'][:]
                # with open(os.path.join(video_dir, 'ground_truth.txt'), 'r') as f:
                #     data = f.readlines()
                # data_timestamp = np.array([float(strr.replace('e','E')) for strr in list(data[2].split())]) # Per-frame timestamps in seconds; start timestamp is 0
                # data_Hr = np.array([float(strr.replace('e','E')) for strr in list(data[1].split())])
                # data_ecg = np.array([float(strr.replace('e','E')) for strr in list(data[0].split())])
                # assert len(data_timestamp) == len(data_ecg)
                # assert len(data_timestamp) == len(frame_path)

                #time_diffs = np.diff(data_timestamp)
                #ecg_rate = 1 / time_diffs.mean()
                ecg_rate = 256.0
                #print(ecg_rate)
                #chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
                #---
                frame_rate = 20.0
                ecg_chunk_split = self.calculate_chunk_split(len(data_ecg), 256.0, self.transform_rate, self.T)
                frame_chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
                # print(ecg_chunk_split)
                # print()
                # print(frame_chunk_split)
                # print(f"ecg_rate: {ecg_rate},ecg length:{len(ecg)} frame_rate: {frame_rate}, frame length: {len(frame_path)}")
                # ecg_rate: 59.58411788057052,ecg length:4047 frame_rate: 29.999992496484868, frame length: 2039
                assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"
                for idx in range(len(ecg_chunk_split)):
                    start_idx, end_idx = ecg_chunk_split[idx]
                    ecg_select = data_ecg[start_idx:end_idx]
                    start_idx, end_idx = frame_chunk_split[idx]
                    # print(idx, start_idx, end_idx)
                    frame_select = frame_path[start_idx:end_idx]
                    #print()
                    self.data_list.append(
                        {'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': ecg_rate})


# Only use samples with lux >= 10
class BUAA(BaseDataset):
    def __init__(self, data_dir, train=True, T=-1, transform_rate=30, limit_batches=0, notAll=True):
        super().__init__(data_dir, train, T, transform_rate, limit_batches=limit_batches, notAll=notAll)

    def get_data_list(self):
        subject_list = os.listdir(self.data_dir)  # "/data/public_dataset/BUAA-MIHR"
        subject_list.sort()

        if self.notAll:
            if self.train:
                subject_list = [subject_list[i] for i in range(10)]

            else:
                subject_list = [subject_list[i] for i in range(10, 13)]
        for subject in subject_list:
            sub_dir = os.path.join(self.data_dir, subject)

            lux_dir = os.listdir(os.path.join(self.data_dir, sub_dir))
            lux_dir = [i for i in lux_dir if float(i[3:]) >= 10]  # lux >= 10
            for lux in lux_dir:
                video_dir = os.path.join(self.data_dir, sub_dir, lux)
                #print(sub_dir,video_dir)
                video_dir_list = os.listdir(video_dir)
                video_name = [i for i in video_dir_list if "avi" in i][0]
                hr_name = video_name.split(".avi")[0] + ".csv"
                ppg_name = video_name.split(".avi")[0] + "_wave.csv"

                #print(subject,lux,video_name,hr_name,ppg_name)

                videoCap = cv2.VideoCapture(os.path.join(video_dir, video_name))
                video_len = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fr = videoCap.get(cv2.CAP_PROP_FPS)

                frame_list_int = [int(i) for i in range(video_len)]
                frame_list_int.sort()  # Convert to integers to ensure correct ordering of frame filenames
                frame_path = []
                mao_path = video_dir.replace("public_dataset", "maoguanhui")
                for frame_name in frame_list_int:
                    #print(mao_path,subject,lux)
                    #print(os.path.join(mao_path,'pic',f"{frame_name}.png"))
                    frame_path.append(os.path.join(mao_path, 'pic', f"{frame_name}.png"))
                # read label
                # print(os.path.join(video_dir,  video_name,f"{frame_name}.png"))

                # df = pd.read_csv(os.path.join(video_dir,hr_name))
                # data_Hr = df["PULSE"]

                df = pd.read_csv(os.path.join(video_dir, ppg_name))
                data_ecg = np.array(df).reshape(-1)

                #print(len(data_Hr) , len(data_ecg),video_len,video_fr)
                ppg = data_ecg  # segNum*160*2 — doubling factor considered from frame-count perspective
                x = np.linspace(1, len(ppg), len(ppg))  # 160/30*60=160*2
                #print(len(ppg),len(x),len(os.listdir(tempPath + '/pic')),segNum) #3817 3840 1920
                funcInterpolate = interpolate.interp1d(x, ppg, kind="slinear")

                xNew = np.linspace(1, len(ppg), 1800)
                data_ecg = funcInterpolate(xNew)

                # print(len(data_Hr) , len(data_ecg),video_len,video_fr)
                # print("----------")

                # Sub 02 lux 39.8 lux39.8_GDB.avi lux39.8_GDB.csv lux39.8_GDB_wave.csv
                # 62 3687 1800 30.0
                # data_timestamp = np.array([float(strr.replace('e','E')) for strr in list(data[2].split())]) # Per-frame timestamps in seconds; start timestamp is 0
                # data_Hr = np.array([float(strr.replace('e','E')) for strr in list(data[1].split())])
                # data_ecg = np.array([float(strr.replace('e','E')) for strr in list(data[0].split())])
                # assert len(data_timestamp) == len(data_Hr) == len(data_ecg)
                # assert len(data_timestamp) == len(frame_path)
                # time_diffs = np.diff(data_timestamp)
                # frame_rate = 1 / time_diffs.mean()
                frame_rate = 30.0
                chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
                for start_idx, end_idx in chunk_split:
                    frame_select = frame_path[start_idx:end_idx]
                    ecg_select = data_ecg[start_idx:end_idx]
                    self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate,
                                           'ecg_rate': frame_rate})


class VIPL(BaseDataset):

    def get_data_list(self):
        self.name = "VIPL"
        self.fold_split_dir = os.path.join(self.data_dir, "VIPL_fold")
        self.fold_list = []
        for i in range(1, 6):
            mat_path = os.path.join(self.fold_split_dir, f"fold{i}.mat")
            mat = sio.loadmat(mat_path)
            # print(mat[f"fold{i}"])
            self.fold_list.append(mat[f"fold{i}"].reshape(-1))
        fold = np.hstack(self.fold_list)
        if self.notAll:
            if self.train:
                fold = np.hstack((*self.fold_list[:self.kfold - 1], *self.fold_list[self.kfold:]))
            else:
                fold = self.fold_list[self.kfold - 1]
        p_lists = [f"p{i}" for i in fold]
        p_lists.sort()
        video_path = self.data_dir
        for p_name in p_lists:
            p_root = os.path.join(video_path, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                # print(v_root)

                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                if "source3No_5.png" in source_lists:
                    source_lists.remove("source3No_5.png")

                source_lists.sort()
                for source_name in source_lists:
                    # read video
                    pic_type = 'align_crop_pic'
                    frame_dir = os.path.join(v_root, source_name, pic_type)
                    # print(source_name)
                    if frame_dir in [f'{video_path}/p45/v1/source2/{pic_type}',
                                     f'{video_path}/p32/v7/source3/{pic_type}',
                                     f'{video_path}/p27/v9/source2/{pic_type}',
                                     f'{video_path}/p28/v5/source3/{pic_type}',
                                     f'{video_path}/p48/v7/source1/{pic_type}',
                                     f'{video_path}/p96/v3/source3/{pic_type}',
                                     f"{video_path}/p71/v6/source3/{pic_type}",
                                     f"{video_path}/p10/v1/source1/{pic_type}"]:
                        continue
                    frame_list = os.listdir(frame_dir)
                    if len(frame_list) == 0:
                        print(f"empty frame_dir: {frame_dir}")
                        continue
                    try:
                        frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                        frame_list_int.sort()  # Convert to integers to ensure correct ordering of frame filenames
                    except:
                        print(f"frame path sorted wrong in {frame_dir}")
                        continue
                    frame_path = []
                    seg_dir = os.path.join(v_root, source_name, "seg")
                    seg_path = []
                    shading_dir = os.path.join(v_root, source_name, "shading")
                    shading_path = []
                    for frame_name in frame_list_int:
                        frame_path.append(os.path.join(frame_dir, f"{frame_name:05d}.png"))
                        seg_path.append(os.path.join(seg_dir, f"{frame_name:05d}.png"))
                        shading_path.append(os.path.join(shading_dir, f"{frame_name:05d}.png"))
                    # read label
                    # print(frame_path[0])
                    gt_HR_csv = os.path.join(v_root, source_name, "gt_HR.csv")

                    #gt_SpO2_csv = os.path.join(v_root, source_name, "gt_SpO2.csv")
                    wave_csv = os.path.join(v_root, source_name, "wave.csv")
                    with open(wave_csv, 'r') as f:
                        data = f.readlines()
                        data = data[1:]
                        ecg = np.array([int(i) for i in data])

                    rate_txt = os.path.join(v_root, source_name, "rate.txt")
                    with open(rate_txt, 'r') as f:
                        data = f.read().splitlines()
                        frame_rate = float(data[0].split(":")[1])
                        total_frame = float(data[1].split(":")[1])
                        total_time = float(data[2].split(":")[1])

                    ecg_rate = len(ecg) / total_time
                    # print(f"ecg_rate: {ecg_rate},ecg length:{len(ecg)} frame_rate: {frame_rate}, frame length: {len(frame_path)}")
                    try:  # It's possible that after resampling to the target frame rate there are not enough frames
                        ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, self.transform_rate, self.T)
                        frame_chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate,
                                                                       self.T)
                    except:
                        continue
                    # print(ecg_chunk_split)
                    # print()
                    # print(frame_chunk_split)
                    split_length = min(len(ecg_chunk_split), len(frame_chunk_split))
                    ecg_chunk_split = ecg_chunk_split[:split_length]
                    frame_chunk_split = frame_chunk_split[:split_length]
                    # print(f"ecg_rate: {ecg_rate},ecg length:{len(ecg)} frame_rate: {frame_rate}, frame length: {len(frame_path)}")
                    assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"
                    for idx in range(len(ecg_chunk_split)):
                        start_idx, end_idx = ecg_chunk_split[idx]
                        ecg_select = ecg[start_idx:end_idx]
                        start_idx, end_idx = frame_chunk_split[idx]
                        # print(idx, start_idx, end_idx)
                        frame_select = frame_path[start_idx:end_idx]
                        seg_select = seg_path[start_idx:end_idx]
                        shading_select = shading_path[start_idx:end_idx]
                        self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, "frame_rate": frame_rate,
                                               "ecg_rate": ecg_rate, \
                                               "location": f"{p_name}_{v_name}_{source_name}_{start_idx}_{end_idx}",
                                               "clip_idx": idx, \
                                               "seg_path": seg_select, "shading_path": shading_select})
        #         break
        #     break


class VIPL_yzt(BaseDataset):
    def get_data_list(self):
        self.name = "VIPL"
        data_dir = self.data_dir
        self.sample_dict = {}
        for train_data in ['VIPL_ECCV_train1', 'VIPL_ECCV_train2']:
            with open(f'{data_dir}/train_val_list/{train_data}.txt', 'r') as f:
                data = f.readlines()
            for line in data:
                sample_name = line.split()[0]
                start_frame = int(line.split()[1]) - 1
                bvp_rate = float(line.split()[2])
                clip_average_HR = float(line.split()[3])
                bvp_signals = np.array([float(i) for i in line.split()[5:5 + 160]])
                if sample_name not in self.sample_dict:
                    self.sample_dict[sample_name] = {start_frame: {'bvp_rate': bvp_rate, 'bvp_signals': bvp_signals,
                                                                   "clip_average_HR": clip_average_HR}}
                elif start_frame not in self.sample_dict[sample_name]:
                    self.sample_dict[sample_name][start_frame] = {'bvp_rate': bvp_rate, 'bvp_signals': bvp_signals,
                                                                  "clip_average_HR": clip_average_HR}
                else:
                    continue

        self.fold_split_dir = os.path.join(self.data_dir, "VIPL_fold")
        self.fold_list = []

        for i in range(1, 6):
            mat_path = os.path.join(self.fold_split_dir, f"fold{i}.mat")
            mat = sio.loadmat(mat_path)
            self.fold_list.append(mat[f"fold{i}"].reshape(-1))

        if self.train:
            # all flod except self.fold
            fold = np.concatenate(self.fold_list[:self.kfold - 1] + self.fold_list[self.kfold:])
        else:
            fold = self.fold_list[self.kfold - 1]

        # print(fold)
        p_lists = [f"p{i}" for i in fold]
        p_lists.sort()
        for p_name in p_lists:
            p_root = os.path.join(self.data_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                source_lists.sort()
                for source_name in source_lists:
                    # read video
                    pic_type = 'align_crop_pic'  # NOTE: pic, align_crop_pic, aligned_pic
                    frame_dir = os.path.join(v_root, source_name, pic_type)
                    if frame_dir in [f'{self.data_dir}/p32/v7/source3/{pic_type}']:
                        continue
                    sample_name = f'{p_name}/{v_name}/{source_name}'
                    if sample_name not in self.sample_dict:
                        raise ValueError(f'{sample_name} not in sample_dict')

                    start_frame = min(self.sample_dict[sample_name].keys())
                    end_frame = 160 * len(self.sample_dict[sample_name].keys()) + start_frame
                    all_ecg_signals = np.concatenate([self.sample_dict[sample_name][sidx]['bvp_signals'] for sidx in
                                                      range(start_frame, end_frame, 160)])
                    if self.T == -1:
                        data_item = {}
                        data_item['frame_path'] = [os.path.join(frame_dir, f"{i:0>5}.png") for i in
                                                   range(start_frame, end_frame)]
                        data_item['ecg'] = all_ecg_signals
                        data_item['frame_rate'] = 30
                        data_item['ecg_rate'] = self.sample_dict[sample_name][start_frame]['bvp_rate']
                        data_item["location"] = sample_name
                        self.data_list.append(data_item)
                    else:
                        for start_frame_iter in range(start_frame, end_frame - len(all_ecg_signals) % self.T, self.T):
                            data_item = {}
                            data_item['frame_path'] = [os.path.join(frame_dir, f"{start_frame_iter + i:0>5}.png") for i
                                                       in range(self.T)]
                            data_item['ecg'] = all_ecg_signals[
                                               start_frame_iter - start_frame: start_frame_iter - start_frame + self.T]
                            data_item['frame_rate'] = 30
                            data_item['ecg_rate'] = self.sample_dict[sample_name][start_frame]['bvp_rate']
                            data_item["location"] = f'{sample_name}_{start_frame_iter}_{start_frame_iter + self.T}'
                            self.data_list.append(data_item)
        # print(len(self.data_list))


class VIPL_yzt2(BaseDataset):
    def get_data_list(self):
        self.name = "VIPL"
        data_dir = self.data_dir
        self.sample_dict = {}

        if self.train:
            fold_list = \
                ['VIPL_ECCV_train1', 'VIPL_ECCV_train2', 'VIPL_ECCV_train3', 'VIPL_ECCV_train4', 'VIPL_ECCV_train5'][
                    self.kfold - 1]
        else:
            fold_list = ['VIPL_ECCV_test1', 'VIPL_ECCV_test2', 'VIPL_ECCV_test3', 'VIPL_ECCV_test4', 'VIPL_ECCV_test5'][
                self.kfold - 1]
        if self.train:
            assert self.T == 160, "VIPL dataset T must be 160 when training"
            with open(f"{data_dir}/train_val_list/{fold_list}.txt", 'r') as f:
                data = f.readlines()
            for line in data:
                sample_name = line.split()[0]
                start_frame = int(line.split()[1]) - 1
                bvp_rate = float(line.split()[2])
                clip_average_HR = float(line.split()[3])
                bvp_signals = np.array([float(i) for i in line.split()[5:5 + 160]])

                p_name, v_name, source_name = sample_name.split('/')
                frame_dir = os.path.join(data_dir, p_name, v_name, source_name, 'align_crop_pic')

                if frame_dir in [f'{self.data_dir}/p32/v7/source3/align_crop_pic']:
                    continue
                sample_name = f'{p_name}/{v_name}/{source_name}'
                seg_dir = os.path.join(data_dir, p_name, v_name, source_name, 'seg')
                shading_dir = os.path.join(data_dir, p_name, v_name, source_name, 'shading')
                frame_list = os.listdir(frame_dir)
                if len(frame_list) == 0:
                    continue
                data_item = {}
                data_item['frame_path'] = [os.path.join(frame_dir, f"{i:0>5}.png") for i in
                                           range(start_frame, start_frame + 160)]
                data_item['ecg'] = bvp_signals
                data_item['frame_rate'] = 30
                data_item['ecg_rate'] = bvp_rate
                data_item["location"] = sample_name
                data_item["seg_path"] = [os.path.join(seg_dir, f"{i:0>5}.png") for i in
                                         range(start_frame, start_frame + 160)]
                data_item["shading_path"] = [os.path.join(shading_dir, f"{i:0>5}.png") for i in
                                             range(start_frame, start_frame + 160)]
                data_item["start_frame"] = start_frame
                self.data_list.append(data_item)
        else:
            raise NotImplementedError("VIPL dataset test not implemented")
            # assert self.T != -1, "VIPL dataset T must not be -1 when testing"
            # with open(f"{data_dir}/train_val_list/{fold_list}.txt", 'r') as f:
            #     data = f.readlines()
            # for line in data:
            #     sample_name = line.split()[0]
            #     total_clips = int(line.split()[1])
            #     # start_frame = int(line.split()[1]) - 1
            #     bvp_rate = float(line.split()[2])
            #     clip_average_HR = float(line.split()[3])
            #     bvp_signals = np.array([float(i) for i in line.split()[5:5+160]])

            #     p_name, v_name, source_name = sample_name.split('/')
            #     frame_dir = os.path.join(data_dir, p_name, v_name, source_name, 'align_crop_pic')

            #     if frame_dir in [f'{self.data_dir}/p32/v7/source3/align_crop_pic']:
            #         continue
            #     sample_name = f'{p_name}/{v_name}/{source_name}'
            #     seg_dir = os.path.join(data_dir, p_name, v_name, source_name, 'seg')
            #     shading_dir = os.path.join(data_dir, p_name, v_name, source_name, 'shading')
            #     frame_list = os.listdir(frame_dir)
            #     if len(frame_list) == 0:
            #         continue
            #     data_item = {}
            #     data_item['frame_path'] = [os.path.join(frame_dir, f"{i:0>5}.png") for i in range(start_frame, start_frame + 160)]
            #     data_item['ecg'] = bvp_signals
            #     data_item['frame_rate'] = 30
            #     data_item['ecg_rate'] = bvp_rate
            #     data_item["location"] = sample_name
            #     data_item["seg_path"] = [os.path.join(seg_dir, f"{i:0>5}.png") for i in range(start_frame, start_frame + 160)]
            #     data_item["shading_path"] = [os.path.join(shading_dir, f"{i:0>5}.png") for i in range(start_frame, start_frame + 160)]
            #     data_item["start_frame"] = start_frame
            #     self.data_list.append(data_item)

        return


class VIPL_test(Dataset):
    def __init__(self, data_dir, T=300, train=False, kfold=1, norm_type="reconstruct"):
        self.root_dir = data_dir
        self.clip_frames = 220  #+60#T
        # assert self.clip_frames == 300 or self.clip_frames == 160, "VIPL dataset T must be 160 or 300"

        self.train = train
        fold_list = ['VIPL_ECCV_test1', 'VIPL_ECCV_test2', 'VIPL_ECCV_test3', 'VIPL_ECCV_test4', 'VIPL_ECCV_test5'][
            kfold - 1]
        self.landmarks_frame = pd.read_csv(f"{data_dir}/train_val_list/{fold_list}.txt", delimiter=' ', header=None)
        self.norm_type = norm_type

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        video_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]), "align_crop_pic")
        sample_name = self.landmarks_frame.iloc[idx, 0]
        total_clips = self.landmarks_frame.iloc[idx, 1]

        video_x = self.get_single_video_x(video_path, total_clips)

        framerate = self.landmarks_frame.iloc[idx, 2]

        clip_average_HR = self.landmarks_frame.iloc[idx, 3]

        seg_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]), "seg")
        shading_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]), "shading")
        seg_video = self.get_single_video_x(seg_path, total_clips, channel=1)
        shading_video = self.get_single_video_x(shading_path, total_clips, channel=1)
        video_x = video_x.transpose(0, 4, 1, 2, 3)
        seg_video = seg_video.squeeze(-1)
        shading_video = shading_video.squeeze(-1)
        sample = {'video': video_x, 'framerate': framerate, 'clip_average_HR_peaks': clip_average_HR,
                  "location": sample_name}

        # if self.transform:
        #     sample = self.transform(sample)
        if self.norm_type == "reconstruct":
            sample['video'] = sample['video'] / 255.0
            sample['seg_video'] = seg_video / 255.0
            sample['shading_video'] = shading_video / 255.0
        elif self.norm_type == "normalize":
            sample['video'] = (sample['video'] - 127.5) / 127.5
            sample['seg_video'] = (seg_video - 127.5) / 127.5
            sample['shading_video'] = (shading_video - 127.5) / 127.5
        return sample

    def get_single_video_x(self, video_path, total_clips, channel=3):
        video_jpgs_path = video_path

        video_x = np.zeros((total_clips, self.clip_frames, 64, 64, channel))
        # print(f"video length: {len(os.listdir(video_jpgs_path))}, clip num: {total_clips} use frame num: {total_clips*160 + 60}")
        # print(video_jpgs_path, total_clips, self.clip_frames, total_clips*self.clip_frames, len(os.listdir(video_jpgs_path)))
        # assert len(os.listdir(video_jpgs_path)) >= total_clips * self.clip_frames, "video frames not equal"
        for tt in range(total_clips):
            image_id = tt * 160 + 61
            for i in range(self.clip_frames):
                s = "%05d" % (image_id - 1)
                image_name = s + '.png'

                # face video
                image_path = os.path.join(video_jpgs_path, image_name)

                if channel == 3:
                    tmp_image = cv2.imread(image_path)
                    try:
                        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
                    except:
                        assert False, f"frame path: {image_path} is wrong"
                    tmp_image = cv2.resize(tmp_image, (64, 64), interpolation=cv2.INTER_CUBIC)
                else:
                    tmp_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    tmp_image = cv2.resize(tmp_image, (64, 64), interpolation=cv2.INTER_CUBIC)
                    tmp_image = tmp_image[:, :, np.newaxis]

                #if tmp_image is None:    # It seems some frames missing
                #    tmp_image = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')

                # tmp_image = cv2.resize(tmp_image, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                video_x[tt, i, :, :, :] = tmp_image

                image_id += 1

        return video_x


class VIPL_yzt3(BaseDataset):
    def get_data_list(self):
        self.name = "VIPL"
        data_dir = self.data_dir
        # p_lists
        if self.train:
            fold_list = \
                ['VIPL_ECCV_train1', 'VIPL_ECCV_train2', 'VIPL_ECCV_train3', 'VIPL_ECCV_train4', 'VIPL_ECCV_train5'][
                    self.kfold - 1]
        else:
            fold_list = ['VIPL_ECCV_test1', 'VIPL_ECCV_test2', 'VIPL_ECCV_test3', 'VIPL_ECCV_test4', 'VIPL_ECCV_test5'][
                self.kfold - 1]

        self.sample_dict = {}
        with open(f'{data_dir}/train_val_list/{fold_list}.txt', 'r') as f:
            data = f.readlines()
        p_lists = []
        for line in data:
            sample_name = line.split()[0]
            p_name, v_name, source_name = sample_name.split('/')
            if p_name not in p_lists:
                p_lists.append(p_name)
        p_lists.sort()
    # Original data
        self.sample_dict = {}
        for train_data in ['VIPL_ECCV_train1', 'VIPL_ECCV_train2']:
            with open(f'{data_dir}/train_val_list/{train_data}.txt', 'r') as f:
                data = f.readlines()
            for line in data:
                sample_name = line.split()[0]
                start_frame = int(line.split()[1]) - 1
                bvp_rate = float(line.split()[2])
                clip_average_HR = float(line.split()[3])
                bvp_signals = np.array([float(i) for i in line.split()[5:5 + 160]])
                if sample_name not in self.sample_dict:
                    self.sample_dict[sample_name] = {start_frame: {'bvp_rate': bvp_rate, 'bvp_signals': bvp_signals,
                                                                   "clip_average_HR": clip_average_HR}}
                elif start_frame not in self.sample_dict[sample_name]:
                    self.sample_dict[sample_name][start_frame] = {'bvp_rate': bvp_rate, 'bvp_signals': bvp_signals,
                                                                  "clip_average_HR": clip_average_HR}
                else:
                    continue

    # Generate data from the original data and p_lists
        for p_name in p_lists:
            p_root = os.path.join(self.data_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                source_lists.sort()
                for source_name in source_lists:
                    # read video
                    pic_type = 'align_crop_pic'  # NOTE: pic, align_crop_pic, aligned_pic
                    frame_dir = os.path.join(v_root, source_name, pic_type)
                    if frame_dir in [f'{self.data_dir}/p32/v7/source3/{pic_type}']:
                        continue
                    sample_name = f'{p_name}/{v_name}/{source_name}'
                    if sample_name not in self.sample_dict:
                        raise ValueError(f'{sample_name} not in sample_dict')

                    start_frame = min(self.sample_dict[sample_name].keys())
                    end_frame = 160 * len(self.sample_dict[sample_name].keys()) + start_frame
                    all_ecg_signals = np.concatenate([self.sample_dict[sample_name][sidx]['bvp_signals'] for sidx in
                                                      range(start_frame, end_frame, 160)])

                    gt_HR_csv = os.path.join(v_root, source_name, "gt_HR.csv")
                    with open(gt_HR_csv, 'r') as f:
                        data = f.readlines()
                        data = data[1:]
                        hr = np.array([int(i) for i in data])
                    if self.T == -1:
                        data_item = {}
                        data_item['frame_dir'] = frame_dir
                        data_item["seg_dir"] = os.path.join(v_root, source_name, "seg")
                        data_item["shading_dir"] = os.path.join(v_root, source_name, "shading")
                        data_item['ecg'] = all_ecg_signals
                        data_item['frame_rate'] = 30
                        data_item['ecg_rate'] = 30
                        data_item["cal_hr_fps"] = self.sample_dict[sample_name][start_frame]['bvp_rate']
                        data_item["start_frame"] = start_frame
                        data_item['min_frame'] = start_frame  # first value is 60
                        data_item['max_frame'] = end_frame
                        data_item["location"] = sample_name
                        data_item["clip_average_HR"] = np.mean(hr)
                        self.data_list.append(data_item)
                    else:
                        for start_frame_iter in range(start_frame, end_frame - len(all_ecg_signals) % self.T, self.T):
                            data_item = {}
                            data_item['frame_dir'] = frame_dir
                            data_item["seg_dir"] = os.path.join(v_root, source_name, "seg")
                            data_item["shading_dir"] = os.path.join(v_root, source_name, "shading")
                            data_item['ecg'] = all_ecg_signals
                            data_item['frame_rate'] = 30
                            data_item['ecg_rate'] = 30
                            data_item['cal_hr_fps'] = self.sample_dict[sample_name][start_frame]['bvp_rate']
                            data_item["start_frame"] = start_frame_iter
                            data_item['min_frame'] = start_frame  # minimum frame index of the whole video
                            data_item['max_frame'] = end_frame  # maximum frame index of the whole video
                            data_item["location"] = f'{sample_name}_{start_frame_iter}_{start_frame_iter + self.T}'
                            data_item["clip_average_HR"] = self.sample_dict[sample_name][start_frame_iter][
                                'clip_average_HR'] if self.T == 160 else np.mean(hr)
                            self.data_list.append(data_item)

    def get_path(self, file_dir, start_frame, end_frame):
        return [os.path.join(file_dir, f"{i:0>5}.png") for i in range(start_frame, end_frame)]

    def get_bvp(self, bvp, start_frame, end_frame):
        return bvp[start_frame - 60:end_frame - 60]

    def apply_transformations(self, video_path, sample, idcs, augment=True):
        speed = 1.0
        clip = read_video(video_path).astype("float32").transpose(3, 0, 1, 2)  # c, t, h, w
        if self.norm_type == "reconstruct":
            seg_path = self.get_path(sample["seg_dir"], sample["start_frame"], sample["start_frame"] + self.T)
            seg_video = np.expand_dims(read_video_2d(seg_path).astype("float32"), 0)
            shading_path = self.get_path(sample["shading_dir"], sample["start_frame"], sample["start_frame"] + self.T)
            shading_video = np.expand_dims(read_video_2d(shading_path).astype("float32"), 0)
        else:
            seg_path = None
            seg_video = None
            shading_path = None
            shading_video = None
        idcs_c = idcs.copy()
        if augment and self.train:
            assert self.train, "augment is only for training"  # therefore T is guaranteed not to be -1
            ## Time resampling
            if self.aug_speed:
                start_idx = sample["min_frame"]
                end_idx = sample["max_frame"]
                entire_video_path = self.get_path(sample["frame_dir"], start_idx, end_idx)

                # freely change speed
                clip, idcs_c, speed = transforms.augment_speed(entire_video_path, idcs, self.T, 3, self.speed_slow,
                                                               self.speed_fast)
                #
                # p = random.random()
                # if p < 0.5:
                #     clip_HR = sample["clip_average_HR"]
                #     if clip_HR > 88:

                # clip, idcs_c, speed = transforms.augment_speed_c(entire_video_path, idcs, self.T, 3, speed)
                if self.norm_type == "reconstruct":
                    seg_path = self.get_path(sample["seg_dir"], start_idx, end_idx)
                    seg_video = transforms.augment_speed_c(seg_path, idcs, self.T, 1, speed)[0]
                    shading_path = self.get_path(sample["shading_dir"], start_idx, end_idx)
                    shading_video = transforms.augment_speed_c(shading_path, idcs, self.T, 1, speed)[0]
            ## Randomly horizontal flip
            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)
                if self.norm_type == "reconstruct":
                    seg_video = transforms.augment_horizontal_flip(seg_video)
                    shading_video = transforms.augment_horizontal_flip(shading_video)

            ## Randomly reverse time
            if self.aug_reverse:
                clip = transforms.augment_time_reversal(clip)
                if self.norm_type == "reconstruct":
                    seg_video = transforms.augment_time_reversal(seg_video)
                    shading_video = transforms.augment_time_reversal(shading_video)

            ## Illumination noise
            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)

            ## Gaussian noise for every pixel
            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)

            ## Random resized cropping
            if self.aug_resizedcrop:
                clip = transforms.random_resized_crop(clip)
                if self.norm_type == "reconstruct":
                    seg_video = transforms.random_resized_crop(seg_video)
                    shading_video = transforms.random_resized_crop(shading_video)

        if self.norm_type == "normalize":
            clip = torch.from_numpy(clip).float()
            clip = clip.add_(-127.5).div_(127.5)
        elif self.norm_type == "reconstruct":
            clip, seg_video, shading_video = torch.from_numpy(clip).float(), torch.from_numpy(
                seg_video).float(), torch.from_numpy(shading_video).float()
            clip = clip.div_(255.0).pow_(2.4)
            seg_video = seg_video.div_(255.0)
            shading_video = shading_video.div_(255.0)
        return clip, seg_video, shading_video, idcs_c, speed

    def get_sample(self, index):
        """return:
        video: (c, t, h, w),
        ecg: (t,)
        clipAverageHR: value
        seg: (t, h, w), 0~1
        shading: (t, h, w), 0~1
        location: name_frame_path
        """
        location = self.data_list[index]["location"]
        start_idx = self.data_list[index]["start_frame"]
        frame_rate = self.data_list[index]["frame_rate"]
        ecg_rate = self.data_list[index]["ecg_rate"]
        cal_hr_fps = self.data_list[index]["cal_hr_fps"]

        idcs = np.arange(start_idx, start_idx + self.T, dtype=int)
        if self.T == -1:
            video_path = self.get_path(self.data_list[index]["frame_dir"], self.data_list[index]["min_frame"],
                                       self.data_list[index]["max_frame"])
        else:
            video_path = self.get_path(self.data_list[index]["frame_dir"], start_idx, start_idx + self.T)
        video, seg_video, shading_video, speed_idcs, speed = self.apply_transformations(video_path,
                                                                                        self.data_list[index], idcs,
                                                                                        augment=True)
        # read ecg
        if speed != 1.0:
            min_idx = int(speed_idcs[0])
            max_idx = int(speed_idcs[-1]) + 1
            orig_x = np.arange(min_idx, max_idx, dtype=int) - 60
            orig_wave = self.data_list[index]["ecg"][orig_x]
            ecg = np.interp(speed_idcs - 60, orig_x, orig_wave)
            HR = -1
        else:
            if self.T == -1:
                ecg = self.get_bvp(self.data_list[index]["ecg"], self.data_list[index]["min_frame"],
                                   self.data_list[index]["max_frame"])
            else:
                ecg = self.get_bvp(self.data_list[index]["ecg"], start_idx, start_idx + self.T)
            HR = -1

        ecg = (ecg - ecg.min()) / np.std(ecg)
        ecg = torch.from_numpy(ecg).float()

    # # Resampling
        # if not self.can_use_cache:
        #     if self.T != -1:
    #         # At this point resampling target is T
        #         if video.shape[1] != self.T:
        #             video = torch.nn.functional.interpolate(video.unsqueeze(0), size=(self.T, self.h, self.w), mode="trilinear").squeeze(0)
        #         if ecg.shape[0] != self.T:
        #             ecg = torch.nn.functional.interpolate(ecg.view(1, 1, -1), size=(video.shape[1],), mode="linear").reshape(-1)
        #         if self.use_seg and seg_video.shape[0] != self.T:
        #             seg_video = torch.nn.functional.interpolate(seg_video.unsqueeze(0).unsqueeze(0), size=(self.T, self.h, self.w), mode="trilinear").reshape(-1, self.h, self.w)
        #         if self.use_shading and shading_video.shape[0] != self.T:
        #             shading_video = torch.nn.functional.interpolate(shading_video.unsqueeze(0).unsqueeze(0), size=(self.T, self.h, self.w), mode="trilinear").reshape(-1, self.h, self.w)
        #     else:
    #         # At this point resampling according to frame rate
        #         new_T = math.ceil(video.shape[1] * self.transform_rate / frame_rate)
        #         if abs(frame_rate - self.transform_rate) > self.rate_threshold:
        #             video = torch.nn.functional.interpolate(video.unsqueeze(0), size=(new_T, self.h, self.w), mode="trilinear").squeeze(0)
        #         if ecg.shape[0] != video.shape[1]:
        #             ecg = torch.nn.functional.interpolate(ecg.view(1, 1, -1), size=(video.shape[1],), mode="linear").reshape(-1)
        #         if self.use_seg and seg_video.shape[0] != video.shape[1]:
        #             seg_video = torch.nn.functional.interpolate(seg_video.unsqueeze(0).unsqueeze(0), size=(new_T, self.h, self.w), mode="trilinear").reshape(-1, self.h, self.w)
        #         if self.use_shading and shading_video.shape[0] != video.shape[1]:
        #             shading_video = torch.nn.functional.interpolate(shading_video.unsqueeze(0).unsqueeze(0), size=(new_T, self.h, self.w), mode="trilinear").reshape(-1, self.h, self.w)

        # calculate clipAverageHR
        # clipAverageHR,_,_ = cal_hr(ecg.reshape(1,-1))
        clipAverageHR = self.data_list[index]["clip_average_HR"]

        # to one sample dict
        # print(video.shape, ecg.shape, clipAverageHR, location, frame_rate, ecg_rate, cal_hr_fps, speed)
        sample = {"video": video, "ecg": ecg, "clipAverageHR": clipAverageHR, "location": location,
                  "frame_rate": frame_rate, "ecg_rate": ecg_rate, "cal_hr_fps": cal_hr_fps,
                  "speed": speed, }
        if self.norm_type == "reconstruct":
            sample["seg"] = seg_video
            sample["shading"] = shading_video
        return sample


class MMSEHR(BaseDataset):
    def get_data_list(self):
        self.name = "MMSE-HR"

        data_root = self.data_dir
        frame_rate = 25
        ecg_rate = 1000

        test_subject_list = ["F022", "F013", "F014", "F015", "F016", "M010", "M011"]
        subject_list = os.listdir(data_root)
        if self.notAll:
            if self.train:
                subject_list = [subject_name for subject_name in subject_list if subject_name not in test_subject_list]
            else:
                subject_list = test_subject_list
        for subject_name in subject_list:
            subject_path = os.path.join(data_root, subject_name)
            for sample_name in os.listdir(subject_path):
                sample_path = os.path.join(subject_path, sample_name)

                video_path = os.path.join(sample_path, "pic")
                _frame_list = os.listdir(video_path)
                _frame_list.sort(key=lambda x: int(x.split(".")[0]))
                frame_list = [os.path.join(video_path, frame_name) for frame_name in _frame_list]
                seg_path = os.path.join(sample_path, "seg")
                seg_list = [os.path.join(seg_path, frame_name) for frame_name in _frame_list]
                shading_path = os.path.join(sample_path, "shading")
                shading_list = [os.path.join(shading_path, frame_name) for frame_name in _frame_list]

                ecg_path = os.path.join(sample_path, "Pulse Rate_BPM.txt")
                with open(ecg_path, 'r') as f:
                    data = f.read().splitlines()
                    ecg = np.array([float(d) for d in data])
                try:  # It's possible that after resampling to the target frame rate there are not enough frames
                    ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, self.transform_rate, self.T)
                    frame_chunk_split = self.calculate_chunk_split(len(frame_list), frame_rate, self.transform_rate,
                                                                   self.T)
                except:
                    continue
                split_length = min(len(ecg_chunk_split), len(frame_chunk_split))
                ecg_chunk_split = ecg_chunk_split[:split_length]
                frame_chunk_split = frame_chunk_split[:split_length]
                assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"
                for idx in range(len(ecg_chunk_split)):
                    start_idx, end_idx = ecg_chunk_split[idx]
                    ecg_select = ecg[start_idx:end_idx]
                    start_idx, end_idx = frame_chunk_split[idx]
                    frame_select = frame_list[start_idx:end_idx]
                    seg_select = seg_list[start_idx:end_idx]
                    shading_select = shading_list[start_idx:end_idx]
                    self.data_list.append(
                        {'frame_path': frame_select, 'ecg': ecg_select, "frame_rate": frame_rate, "ecg_rate": ecg_rate, \
                         "location": f"{subject_name}_{sample_name}_{start_idx}_{end_idx}", "clip_idx": idx, \
                         "seg_path": seg_select, "shading_path": shading_select})


class rPPGAll(Dataset):
    def __init__(self, removed):  # Require: first index must be less than second index
        self.removed = removed

        self.UBFC = UBFC("/data/zhangyizhu/UBFC/", train=True, T=256, transform_rate=30)
        self.PURE = PURE("/data/zhangyizhu/pure/", train=True, T=256, transform_rate=30)
        #dataset = COHFACE("/data/zhangyizhu/cohface/", train=True, T=160, transform_rate=30,notAll = False)
        self.BUAA = BUAA("/data/public_dataset/BUAA-MIHR", train=True, T=256, transform_rate=30, notAll=False)
        self.VIPL = VIPL('/data/maoguanhui/vipl_list/', train=True, T=256, transform_rate=30)
        self.V4V = V4V("/data/maoguanhui/Phase1_Training_Validationsets/", train=True, T=256, transform_rate=30)

        self.dataSetList = [self.UBFC, self.PURE, self.BUAA, self.VIPL, self.V4V]  #
        self.name = ["UBFC", "PURE", "BUAA", "VIPL", "V4V"]  #

        del self.dataSetList[removed]
        del self.name[removed]

        self.countList = []
        self.sampleGetIdList = []
        temp = 0
        for data in self.dataSetList:
            self.countList.append(len(data))
            temp += len(data)
            self.sampleGetIdList.append(temp)

        self.sampleCount = sum(self.countList)

        # print(self.countList)
        # print(self.sampleGetIdList)
        # [745, 1131, 8480]
        # [745, 1876, 10356]
        #self.P = [i /self.sampleCount for i in self.countList  ]

    def __len__(self):
        return self.sampleCount

    def __getitem__(self, index):
        for i in range(len(self.sampleGetIdList)):
            if (index < self.sampleGetIdList[i]):
                indx = index - self.sampleGetIdList[i - 1] if i != 0 else index
                #print(i,indx)
                dataReturn = self.dataSetList[i][indx]
                dataReturn['domain_lable'] = i
                if self.name[i] == 'V4V' or self.name[i] == 'VIPL':
                    dataReturn['np'] = "nxp"
                else:
                    dataReturn['np'] = "np"
                #print(dataReturn['ecg'].shape,dataReturn['video'].shape,self.name[i])
                # tmp = [j for j in self.countList]
                # del tmp[i]
                # dataReturn['w'] = sum(tmp) / self.sampleCount
                return dataReturn


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    # dataset = UBFC("/root/datasets/UBFC/crop/", train=True, T=160, transform_rate=30, use_seg=True, use_shading=True, w=128,h=128)
    # print(len(dataset))
    # dataset.create_cache()
    # dataset = PURE( "/root/datasets/pure_new/crop/", train=True, T=256, transform_rate=30, use_seg=True, use_shading=True)
    # dataset.create_cache()
    # # dataset = COHFACE("/data/zhangyizhu/cohface/", train=True, T=256, transform_rate=30,notAll = False)
    # # dataset = BUAA("/data/public_dataset/BUAA-MIHR", train=True, T=256, transform_rate=30,notAll = False)
    dataset = VIPL('/data3/wuruize/VIPL', train=True, notAll=False, T=-1, transform_rate=30, use_seg=False,
                   use_shading=False, cache_dir="/data3/wuruize/cache")
    # dataset = MMSEHR("/data/hezhongtian/MMSE-HR_process/", train=True, notAll = False, T=160, transform_rate=30, use_seg=False, use_shading=False)
    # data = dataset[0]
    # dataset = VIPL_yzt('/data/hezhongtian/VIPL', train=True, T=160, norm_type="normalize", kfold=1)
    # print("len", len(dataset))
    # dataset = VIPL_yzt2('/data/hezhongtian/VIPL', train=True, T=160, norm_type="normalize", kfold=1)
    # dataset = VIPL_test("/data/hezhongtian/VIPL", T=160, train=False, kfold=2)
    print("len", len(dataset))
    data = dataset[0]
    # dataset = VIPL_yzt3('/root/datasets/VIPL', train=False, T=-1, norm_type="reconstruct", kfold=1)
    # dataset = VIPL_test("/data/hezhongtian/VIPL", T=160, train=False, kfold=2)
    # print("len", len(dataset))
    # data = dataset[0]
    # print(data["video"].shape)

    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     print(i, data['location'], data["video"].shape)
    # start = time()
    # train_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)
    # for idx, batch in enumerate(train_loader):
    #     video = batch['video']
    #     # print(video.shape)
    #     # ecg = batch['ecg'].cuda(non_blocking=True)
    #     # shading = batch['shading'].cuda(non_blocking=True)
    #     if idx > 100:
    #         break
    #     # break
    # end = time()
    # print("Finish with:{} second".format(end - start))
    # dataset.create_cache()
    # import multiprocessing as mp
    # from time import time
    # import torch
    # # dataset = BaseDataset("/root/datasets/UBFC/crop.h5", train=True, T=160)
    # # dataset[19]
    # print(f"num of CPU: {mp.cpu_count()}")
    # for num_workers in range(6, mp.cpu_count(), 2):
    #     print(f"num_workers: {num_workers}")
    #     train_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=8, pin_memory=False, num_workers=num_workers)
    #     start = time()
    #     for epoch in range(1, 3):
    #         for i, data in enumerate(train_loader, 0):
    #             pass

    #     end = time()
    #     print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
