from typing import List, Tuple, Union, Optional
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from monai.networks.nets.segresnet import SegResNet
from monai.networks.blocks.aspp import SimpleASPP
from common import correlate, MINDSSC, concat_flow, identity_grid_torch, warp_image


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.convz = nn.Conv3d(hidden_dim + input_dim, hidden_dim, 3, padding="same")
        self.convr = nn.Conv3d(hidden_dim + input_dim, hidden_dim, 3, padding="same")
        self.convq = nn.Conv3d(hidden_dim + input_dim, hidden_dim, 3, padding="same")

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convz(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, correlation_features: int):
        super(BasicMotionEncoder, self).__init__()
        # FIXME: I don't like these hard-coded feature dimensions
        self.convc1 = nn.Conv3d(correlation_features, 8, 1, padding='same')
        self.convc2 = nn.Conv3d(8, 8, 3, padding='same')

        self.convf1 = nn.Conv3d(3, 4, 7, padding='same')
        self.convf2 = nn.Conv3d(4, 8, 3, padding='same')
        self.conv = nn.Conv3d(16, 16-3, 3, padding='same')

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class NoCorrMotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # FIXME: I don't like these hard-coded feature dimensions
        self.convf1 = nn.Conv3d(3, 4, 7, padding='same')
        self.convf2 = nn.Conv3d(4, 16-3, 3, padding='same')

    def forward(self, flow):
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        return torch.cat([flo, flow], dim=1)

class UpdateBlockNoCorr(nn.Module):
    def __init__(
        self, hidden_dim: int=32, input_dim=32
    ):
        super().__init__()
        self.encoder = NoCorrMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.flowhead = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim*2, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dim*2, 3, 3, padding='same'),
        )

    def forward(self, flow: torch.Tensor, hidden: torch.Tensor, inp: torch.Tensor):
        mf = self.encoder(flow)
        in_gru = torch.cat([mf, inp], dim=1)
        next_hidden = self.gru(hidden, in_gru)
        delta_flow = self.flowhead(next_hidden)
        return next_hidden, delta_flow


class UpdateBlock(nn.Module):
    def __init__(
        self, correlation_features, hidden_dim: int=32, input_dim=32
    ):
        super().__init__()
        self.encoder = BasicMotionEncoder(correlation_features=correlation_features)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.flowhead = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim*2, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dim*2, 3, 3, padding='same'),
        )


    def forward(self, correlation: torch.Tensor, flow: torch.Tensor, hidden: torch.Tensor, inp: torch.Tensor):
        mf = self.encoder(flow, correlation)
        in_gru = torch.cat([mf, inp], dim=1)
        next_hidden = self.gru(hidden, in_gru)
        delta_flow = self.flowhead(next_hidden)
        return next_hidden, delta_flow

class SomeNetNoCorr(nn.Module):
    def __init__(self, hidden_dim: int=64, input_size: int=16, iters:int=12, diffeomorphic: bool = False):
        super().__init__()
        self.feature_extractor = SegResNet(in_channels=1, out_channels= input_size//2)
        self.context = SegResNet(in_channels=4, out_channels=hidden_dim)

        self.update = UpdateBlockNoCorr(hidden_dim=hidden_dim)
        self.starting = None
        self.flow_upsample = nn.Sequential(
            nn.Conv3d(3,3,3,1,padding='same'),
            nn.ReLU(),
            nn.Conv3d(3,3,3,1,padding='same'),
        )
        self.iters=iters
        self.diffeomorphic = diffeomorphic

    def apply_diffeomorphism(self, flow):
        scale = 1 / (2**7)
        flow = scale * flow
        for _ in range(7):
            flow = flow + concat_flow(flow, flow)
        return flow

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        hidden_init: Optional[torch.Tensor]=None,
        ret_fmap: bool=False
    ):

        if self.starting == None:
            self.starting = torch.zeros((moving.shape[0], 3, *moving.shape[-3:]), device=moving.device)

        hidden = self.context(torch.cat([self.starting,  moving], dim=1))
        fixed_feat = self.feature_extractor(fixed)
        moving_feat = self.feature_extractor(moving)
        inp = torch.cat((fixed_feat, moving_feat), dim=1)
        hidden = torch.tanh(hidden)
        if hidden_init is not None:
            hidden = hidden + hidden_init
        inp = torch.relu(inp)

        delta_flow = self.starting
        flow = delta_flow

        for _ in range(self.iters):
            hidden, delta_flow = self.update(delta_flow, hidden, inp)
            flow = flow + delta_flow

            if self.diffeomorphic:
                self.apply_diffeomorphism(flow)

            moving_ = warp_image(flow, moving)
            moving_feat = self.feature_extractor(moving_)
            inp = torch.cat((fixed_feat, moving_feat), dim=1)

        if not ret_fmap:
            return flow, hidden
        else:
            return flow, hidden, fixed_feat, moving_feat


class SomeNet(nn.Module):
    def __init__(self, search_range: int=3, hidden_dim: int=64, input_size: int=16, iters:int=12, diffeomorphic: bool = False):
        super().__init__()
        self.search_range = search_range
        # self.context= Unet3D(infeats=1, outfeats=80)

        self.feature_extractor = SegResNet(in_channels=1, out_channels= input_size)
        self.context = SegResNet(in_channels=4, out_channels=hidden_dim)

        self.update = UpdateBlock((2*search_range+1)**3, hidden_dim=hidden_dim)
        self.starting = None
        self.flow_upsample = nn.Sequential(
            nn.Conv3d(3,3,3,1,padding='same'),
            nn.ReLU(),
            nn.Conv3d(3,3,3,1,padding='same'),
        )
        self.count = 0
        self.iters=iters
        self.diffeomorphic = diffeomorphic
        self.starting = None #torch.zeros((moving.shape[0], 3, *moving.shape[-3:]), device=moving.device) 

    def compute_correlation(self, fixed_feat: torch.Tensor, moving_feat: torch.Tensor):
        with torch.no_grad():
            correlation, correlation_argmin = correlate(fixed_feat, moving_feat, search_radius=self.search_range)
            correlation = correlation.squeeze().unsqueeze(0)

        return correlation

    def apply_diffeomorphism(self, flow):
        scale = 1 / (2**7)
        flow = scale * flow
        for _ in range(7):
            flow = flow + concat_flow(flow, flow)
        return flow

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        hidden_init: Optional[torch.Tensor]=None,
        ret_fmap: bool=False
    ):
        fixed_feat = self.feature_extractor(fixed)
        moving_feat = self.feature_extractor(moving)

        cost_volume = self.compute_correlation(fixed_feat, moving_feat,)

        if self.starting is None:
            self.starting = torch.zeros((moving.shape[0], 3, *moving.shape[-3:]), device=moving.device) 
            moving = warp_image(self.starting, moving)
            moving_feat = warp_image(self.starting, moving_feat)

        delta_flow = self.starting
        flow = delta_flow

        hidden = self.context(torch.cat([flow,  moving], dim=1))
        hidden = torch.tanh(hidden)
        if hidden_init is not None:
            hidden = hidden + hidden_init

        for _ in range(self.iters):
            cost_volume = self.compute_correlation(fixed_feat, moving_feat)
            hidden, delta_flow = self.update(cost_volume, delta_flow, hidden, moving_feat)
            flow = flow + delta_flow

            if self.diffeomorphic:
                self.apply_diffeomorphism(flow)

            moving_feat = warp_image(delta_flow, moving_feat)

        if not ret_fmap:
            return flow, hidden
        else:
            return flow, hidden, fixed_feat, moving_feat


class SomeNetNoisy(nn.Module):
    def __init__(self, search_range: int=3, hidden_dim: int=64, input_size: int=16, iters:int=12, diffeomorphic: bool = False, noise_prob: float=.7):
        super().__init__()
        self.search_range = search_range
        # self.context= Unet3D(infeats=1, outfeats=80)

        self.feature_extractor = SegResNet(in_channels=1, out_channels= input_size)
        self.context = SegResNet(in_channels=4, out_channels=hidden_dim)

        self.update = UpdateBlock((2*search_range+1)**3, hidden_dim=hidden_dim)
        self.starting = None
        self.flow_upsample = nn.Sequential(
            nn.Conv3d(3,3,3,1,padding='same'),
            nn.ReLU(),
            nn.Conv3d(3,3,3,1,padding='same'),
        )
        self.count = 0
        self.iters=iters
        self.diffeomorphic = diffeomorphic
        self.starting = None
        self.noise_prob = noise_prob

    def compute_correlation(self, fixed_feat: torch.Tensor, moving_feat: torch.Tensor):
        with torch.no_grad():
            correlation, correlation_argmin = correlate(fixed_feat, moving_feat, search_radius=self.search_range)
            correlation = correlation.squeeze().unsqueeze(0)

        return correlation

    def apply_diffeomorphism(self, flow):
        scale = 1 / (2**7)
        flow = scale * flow
        for _ in range(7):
            flow = flow + concat_flow(flow, flow)
        return flow

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        hidden_init: Optional[torch.Tensor]=None,
        ret_fmap: bool=False,
    ):
        fixed_feat = self.feature_extractor(fixed)
        moving_feat = self.feature_extractor(moving)

        cost_volume = self.compute_correlation(fixed_feat, moving_feat,)

        if self.starting is None:
            self.starting = torch.zeros((moving.shape[0], 3, *moving.shape[-3:]), device=moving.device)

        delta_flow = self.starting
        flow = delta_flow

        moving = warp_image(flow, moving)
        moving_feat = warp_image(flow, moving_feat)

        hidden = self.context(torch.cat([flow,  moving], dim=1))
        hidden = torch.tanh(hidden)
        if hidden_init is not None:
            hidden = hidden + hidden_init

        for iter in range(self.iters):
            cost_volume = self.compute_correlation(fixed_feat, moving_feat)
            hidden, delta_flow = self.update(cost_volume, delta_flow, hidden, moving_feat)

            # if self.diffeomorphic:
            #     self.apply_diffeomorphism(flow)

            if self.training and random.random() < self.noise_prob:
                scale = delta_flow.max() - delta_flow.min()
                additive_noise = .5 * scale * torch.randn(1, device=scale.device)
                delta_flow = additive_noise + delta_flow #(delta_flow * torch.rand(1, device=delta_flow.device))
                hidden = warp_image(delta_flow, hidden)

            moving_feat = warp_image(delta_flow, moving_feat)
            flow = flow + delta_flow

        if not ret_fmap:
            return flow, hidden
        else:
            return flow, hidden, fixed_feat, moving_feat

class SomeNetNoisyv2(nn.Module):
    def __init__(self, search_range: int=3, hidden_dim: int=64, input_size: int=16, iters:int=12, diffeomorphic: bool = False):
        super().__init__()
        self.search_range = search_range
        # self.context= Unet3D(infeats=1, outfeats=80)

        self.feature_extractor = SegResNet(in_channels=1, out_channels= input_size)
        self.context = SegResNet(in_channels=4, out_channels=hidden_dim)

        self.update = UpdateBlock((2*search_range+1)**3, hidden_dim=hidden_dim)
        self.starting = None
        self.flow_upsample = nn.Sequential(
            nn.Conv3d(3,3,3,1,padding='same'),
            nn.ReLU(),
            nn.Conv3d(3,3,3,1,padding='same'),
        )
        self.count = 0
        self.iters=iters
        self.diffeomorphic = diffeomorphic
        self.starting = None

    def compute_correlation(self, fixed_feat: torch.Tensor, moving_feat: torch.Tensor):
        with torch.no_grad():
            correlation, correlation_argmin = correlate(fixed_feat, moving_feat, search_radius=self.search_range)
            correlation = correlation.squeeze().unsqueeze(0)

        return correlation

    def apply_diffeomorphism(self, flow):
        scale = 1 / (2**7)
        flow = scale * flow
        for _ in range(7):
            flow = flow + concat_flow(flow, flow)
        return flow

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        hidden_init: Optional[torch.Tensor]=None,
        ret_fmap: bool=False,
    ):
        fixed_feat = self.feature_extractor(fixed)
        moving_feat = self.feature_extractor(moving)

        cost_volume = self.compute_correlation(fixed_feat, moving_feat,)

        if self.starting is None:
            self.starting = torch.zeros((moving.shape[0], 3, *moving.shape[-3:]), device=moving.device)

        delta_flow = self.starting
        flow = delta_flow

        moving = warp_image(flow, moving)
        moving_feat = warp_image(flow, moving_feat)

        hidden = self.context(torch.cat([flow,  moving], dim=1))
        hidden = torch.tanh(hidden)
        if hidden_init is not None:
            hidden = hidden + hidden_init

        for iter in range(self.iters):
            cost_volume = self.compute_correlation(fixed_feat, moving_feat)
            hidden, delta_flow = self.update(cost_volume, delta_flow, hidden, moving_feat)

            if self.training:
                additive_noise = torch.max(delta_flow) * torch.randn_like(delta_flow, device=delta_flow.device)
                noisy_delta = additive_noise + (delta_flow * torch.rand(1, device=delta_flow.device))
                delta_flow = noisy_delta

            moving_feat = warp_image(delta_flow, moving_feat)
            flow = flow + delta_flow

        if not ret_fmap:
            return flow, hidden
        else:
            return flow, hidden, fixed_feat, moving_feat

class SomeNetBetterContext(nn.Module):
    def __init__(self, search_range: int=3, hidden_dim: int=64, input_size: int=16, iters:int=12, diffeomorphic: bool = False):
        super().__init__()
        self.search_range = search_range
        # self.context= Unet3D(infeats=1, outfeats=80)

        self.feature_extractor = SegResNet(in_channels=1, out_channels= input_size)
        # self.context = SegResNet(in_channels=4, out_channels=hidden_dim)
        self.context = SimpleASPP(spatial_dims=3, in_channels=1, conv_out_channels=hidden_dim//4)

        self.update = UpdateBlock((2*search_range+1)**3, hidden_dim=hidden_dim)
        self.starting = None
        self.flow_upsample = nn.Sequential(
            nn.Conv3d(3,3,3,1,padding='same'),
            nn.ReLU(),
            nn.Conv3d(3,3,3,1,padding='same'),
        )
        self.count = 0
        self.iters=iters
        self.diffeomorphic = diffeomorphic
        self.starting = None

    def compute_correlation(self, fixed_feat: torch.Tensor, moving_feat: torch.Tensor):
        with torch.no_grad():
            correlation, correlation_argmin = correlate(fixed_feat, moving_feat, search_radius=self.search_range)
            correlation = correlation.squeeze().unsqueeze(0)

        return correlation

    def apply_diffeomorphism(self, flow):
        scale = 1 / (2**7)
        flow = scale * flow
        for _ in range(7):
            flow = flow + concat_flow(flow, flow)
        return flow

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        hidden_init: Optional[torch.Tensor]=None,
        ret_fmap: bool=False,
        train: bool=False
    ):
        fixed_feat = self.feature_extractor(fixed)
        moving_feat = self.feature_extractor(moving)

        cost_volume = self.compute_correlation(fixed_feat, moving_feat,)

        if self.starting is None:
            self.starting = torch.zeros((moving.shape[0], 3, *moving.shape[-3:]), device=moving.device)

        delta_flow = self.starting
        flow = delta_flow

        moving = warp_image(flow, moving)
        moving_feat = warp_image(flow, moving_feat)

        hidden = self.context(torch.cat([flow,  moving], dim=1))
        hidden = torch.tanh(hidden)
        if hidden_init is not None:
            hidden = hidden + hidden_init

        for iter in range(self.iters):
            cost_volume = self.compute_correlation(fixed_feat, moving_feat)
            hidden, delta_flow = self.update(cost_volume, delta_flow, hidden, moving_feat)

            flow = flow + delta_flow

            # if self.diffeomorphic:
            #     self.apply_diffeomorphism(flow)

            if train:
                noisy_delta = torch.randn_like(delta_flow, device=delta_flow.device) + \
                        (delta_flow * torch.rand(1, device=delta_flow.device))
                moving_feat = warp_image(noisy_delta, moving_feat)
            else:
                moving_feat = warp_image(delta_flow, moving_feat)

        if not ret_fmap:
            return flow, hidden
        else:
            return flow, hidden, fixed_feat, moving_feat
