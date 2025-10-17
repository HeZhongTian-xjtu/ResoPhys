import torch
import torch.nn as nn
tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft

"""Contrastive loss combining spatiotemporal sampling (ST_sampling) and
mean-squared-error (MSE) distance between normalized PSDs."""
class ContrastLoss(nn.Module):
    def __init__(self, delta_t, K, Fs, high_pass, low_pass,flag):
        """
        Args:
            delta_t: temporal length of rPPG samples
            K: number of samples per spatial location
            Fs: sampling frequency used for PSD
            high_pass, low_pass: PSD filter cutoffs
            flag: mode of the loss ('p','n','normal')
        """
        super(ContrastLoss, self).__init__()
        self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass) # spatiotemporal sampler
        self.distance_func = nn.MSELoss(reduction = 'mean') # mean squared error for comparing two PSDs
        self.flag = flag

    def compare_samples(self, list_a, list_b, exclude_same=False):
        """Compute average MSE distance between two lists of PSD samples."""
        if exclude_same:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    if i != j:
                        total_distance += self.distance_func(list_a[i], list_b[j])
                        M += 1
        else:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1
        return total_distance / M

    def forward(self, model_output):
        if(self.flag == 'p'):
            samples = self.ST_sampling(model_output)

            # positive loss
            pos_loss_1 = (self.compare_samples(samples[0], samples[0], exclude_same=True) + self.compare_samples(samples[1], samples[1], exclude_same=True)) / 2
            pos_loss_2 = self.compare_samples(samples[0], samples[1], exclude_same=False)

            # overall contrastive loss
            loss = pos_loss_1 + pos_loss_2
            return loss
        elif(self.flag == 'normal'):
            samples = self.ST_sampling(model_output)

            # positive loss
            pos_loss = (self.compare_samples(samples[0], samples[0], exclude_same=True) + self.compare_samples(samples[1], samples[1], exclude_same=True)) / 2

            # negative loss
            neg_loss = -self.compare_samples(samples[0], samples[1])

            # overall contrastive loss
            loss = pos_loss + neg_loss
            return loss, pos_loss, neg_loss
        elif(self.flag == 'n'):
            samples = self.ST_sampling(model_output)

            # negative loss
            neg_loss = -self.compare_samples(samples[0], samples[1])
            return neg_loss


        # two sets of rPPG samples
        # samples = self.ST_sampling(model_output) # a list with length 2 including rPPG samples from the first video and rPPG samples from the second video
        # samples_ = self.ST_sampling(model_output)

        # We list combinations for both pos. loss (pull rPPG samples from the same video) and neg. loss (repel rPPG samples from two different videos).
        # positive loss
        # pos_loss = (self.compare_samples(samples[0], samples_[0]) + self.compare_samples(samples[1], samples_[1])
        #     + self.compare_samples(samples_[0], samples_[0], exclude_same=True) + self.compare_samples(samples_[1], samples_[1], exclude_same=True)
        #     + self.compare_samples(samples[0], samples[0], exclude_same=True) + self.compare_samples(samples[1], samples[1], exclude_same=True)) / 6
        # # negative loss
        # neg_loss = -(self.compare_samples(samples[0], samples[1]) + self.compare_samples(samples_[0], samples_[1])
        #     + self.compare_samples(samples[0], samples_[1]) + self.compare_samples(samples_[0], samples[1])) / 4

        # # overall contrastive loss
        # loss = pos_loss + neg_loss

        # return overall loss, positive loss, and negative loss


class ST_sampling(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    """
    Args:
        delta_t: time length of each rPPG sample (frames)
        K: number of rPPG samples per spatial location
        Fs: sampling frequency used for PSD
        high_pass, low_pass: band-pass bounds (in BPM) to keep
    """
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    """
    Input: tensor of shape (2, M, T) where 2 indicates two videos, M is the
    spatial dimension and T is temporal length.
    Returns a list of PSD-normalized rPPG sample tensors for each video.
    For each spatial location, K random temporal segments of length delta_t are sampled.
    Each segment is converted to normalized PSD and stored.
    """
    def forward(self, input): # input: (2, M, T)
        samples = []
        for b in range(input.shape[0]): # loop over videos (totally 2 videos)
            samples_per_video = []
            for c in range(input.shape[1]): # loop for sampling over spatial dimension
                for i in range(self.K): # loop for sampling K samples with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    x = self.norm_psd(input[b, c, offset:offset + self.delta_t])
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    """
    Args:
        Fs: sampling frequency (Hz)
        high_pass, low_pass: band-pass filter cutoffs (BPM)
    """
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    """
    Args:
        x: input time-domain signal tensor (rPPG)
        zero_pad: fraction of zero-padding relative to signal length (optional)
    Returns:
        normalized PSD vector for the specified frequency band
    Steps:
    - remove mean (DC)
    - zero-pad if requested to increase FFT resolution
    - compute PSD via real FFT
    - band-pass filter the PSD
    - normalize PSD to sum to 1
    """
    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x