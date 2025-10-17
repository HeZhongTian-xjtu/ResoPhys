import numpy as np
import torch
import torch.nn.functional as F
# from rppg.data.new_dataset import read_video, read_video_2d
import cv2

import numpy as np
import torch
import torch.nn.functional as F
# from rppg.data.new_dataset import read_video, read_video_2d
import cv2
import numpy as np
import torch
import torch.nn.functional as F
# from rppg.data.new_dataset import read_video, read_video_2d
import cv2


# Read a sequence of frames from the given file paths, process them,
# and store them in a NumPy array
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


# Read a sequence of 2D masks (segmentation/mask or similar) from given file paths
def read_video_2d(seg_path):
    # Read seg/mask or other 2D mask images
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


# Resample a video clip to a target temporal length
def resample_clip(video, length):
    video = np.transpose(video, (3,0,1,2)).astype(float)
    video = interpolate_clip(video, length)
    video = np.transpose(video, (1,2,3,0))
    return video


# Rearrange image channels according to the specified order
def arrange_channels(imgs, channels):
    d = {'b':0, 'g':1, 'r':2, 'n':3}
    channel_order = [d[c] for c in channels]
    imgs = imgs[:,:,:,channel_order]
    return imgs


# Prepare clip data and convert it into the required format
def prepare_clip(clip):
    # clip = arrange_channels(clip, channels)
    clip = np.transpose(clip, (3, 0, 1, 2)) # [C,T,H,W]
    clip = clip.astype(np.float64)
    return clip


# Interpolate a clip to frames_per_clip length given slicing indices and
# a random speed sampled from [speed_slow, speed_fast]
def augment_speed(clip, idcs, frames_per_clip, speed_slow, speed_fast):
    ''' Interpolates clip to frames_per_clip length given slicing indices, which
        can be floats.
    '''
    vid_len = len(clip)
    within_bounds = False
    while not within_bounds:
        speed_c = np.random.uniform(speed_slow, speed_fast)
        min_idx = idcs[0].astype(int)
        max_idx = np.round(frames_per_clip * speed_c + min_idx).astype(int)
        if max_idx < vid_len:
            within_bounds = True
    speed_c = (max_idx - min_idx) / frames_per_clip # accomodate rounding of end-indices
    clip = clip[min_idx:max_idx]
    clip = prepare_clip(clip)
    interped_clip = interpolate_clip(clip, frames_per_clip)
    interped_idcs = np.linspace(min_idx, max_idx-1, frames_per_clip)
    return interped_clip, interped_idcs, speed_c


# Interpolate a clip at a specified speed
def augment_speed_c(clip, idcs, frames_per_clip, speed_c):
    ''' Interpolate a clip at a specified speed
    '''
    vid_len = len(clip)
    min_idx = idcs[0].astype(int)
    max_idx = np.round(frames_per_clip * speed_c + min_idx).astype(int)
    clip = clip[min_idx:max_idx]
    clip = prepare_clip(clip)
    interped_clip = interpolate_clip(clip, frames_per_clip)
    interped_idcs = np.linspace(min_idx, max_idx-1, frames_per_clip)
    return interped_clip, interped_idcs, speed_c


# Interpolate a clip in time to the specified temporal length
def interpolate_clip(clip, length):
    '''
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    '''
    clip = torch.from_numpy(clip[np.newaxis])
    clip = F.interpolate(clip, (length, 64, 64), mode='trilinear', align_corners=True)
    return clip[0].numpy()


# Resize the spatial dimensions of a clip, scaling the spatial size while
# preserving the temporal length
def resize_clip(clip, length):
    '''
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: spatial size for H and W in the output
    Returns:
        Tensor of shape [C,T,H,W]
    '''
    T = clip.shape[1]
    clip = torch.from_numpy(np.ascontiguousarray(clip[np.newaxis]))
    clip = F.interpolate(clip, (T, length, length), mode='trilinear', align_corners=False)
    return clip[0].numpy()


# Randomly crop a subregion from the clip and resize it back to original size
def random_resized_crop(clip, crop_scale_lims=[0.5, 1]):
    ''' Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    '''
    C,T,H,W = clip.shape
    crop_scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    crop_length = np.round(crop_scale * H).astype(int)
    crop_start_lim = H - (crop_length)
    x1 = np.random.randint(0, crop_start_lim+1)
    y1 = x1
    x2 = x1 + crop_length
    y2 = y1 + crop_length
    cropped_clip = clip[:,:,y1:y2,x1:x2]
    resized_clip = resize_clip(cropped_clip, H)
    return resized_clip, crop_scale


# Randomly crop a subregion from the clip (using provided crop_scale) and
# resize it back to the original size
def random_resized_crop_crop_scale(clip, crop_scale_lims=[0.5, 1], crop_scale=0.5):
    ''' Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    '''
    C,T,H,W = clip.shape
    # crop_scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    crop_length = np.round(crop_scale * H).astype(int)
    crop_start_lim = H - (crop_length)
    x1 = np.random.randint(0, crop_start_lim+1)
    y1 = x1
    x2 = x1 + crop_length
    y2 = y1 + crop_length
    cropped_clip = clip[:,:,y1:y2,x1:x2]
    resized_clip = resize_clip(cropped_clip, H)
    return resized_clip, crop_scale



# Add Gaussian noise to a clip
def augment_gaussian_noise(clip):
    clip = clip + np.random.normal(0, 2, clip.shape)
    return clip


# Add illumination (brightness) noise to a clip
def augment_illumination_noise(clip):
    clip = clip + np.random.normal(0, 10)
    return clip


# Reverse the temporal order of a clip
def augment_time_reversal(clip):
    # if np.random.rand() > 0.5:
    clip = np.flip(clip, 1)
    return clip


# Flip a clip horizontally
def augment_horizontal_flip(clip):
    # if np.random.rand() > 0.5:
    clip = np.flip(clip, 3)
    return clip

