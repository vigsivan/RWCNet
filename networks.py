from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from common import correlate_grad

from image_similarity_matrices import mind_mse


def default_unet_features() -> List[List[int]]:
    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
    return nb_features


def _conv_bn_lrelu(
    infeats: int, outfeats: int, kernel_size: int, stride: int
) -> nn.Module:
    padding = "same" if stride == 1 else 0
    return nn.Sequential(
        nn.Conv3d(
            in_channels=infeats,
            out_channels=outfeats,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm3d(outfeats),
        nn.LeakyReLU(),
    )


def _compute_pad(
    original_shape: Tuple[int, ...], actual_shape: Tuple[int, ...], stride: int,
) -> Tuple[int, ...]:
    padding = [0] * 2 * len(original_shape)
    for i, (ogs, acs) in enumerate(zip(original_shape, actual_shape)):
        expected = ogs // stride
        if expected > acs:
            half = (expected - acs) // 2
            rem = expected - acs - half
            padding[2 * i] = half
            padding[(2 * i) + 1] = rem

    # NOTE: pytorch accepts padding in reverse order
    return tuple(padding[::-1])


class FeatureExtractor(nn.Module):
    """
    Feature Extractor for flownet model
    """

    def __init__(
        self,
        infeats: int,
        feature_sizes: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()
        if len(feature_sizes) != len(kernel_sizes):
            raise ValueError("Features and kernels list must be the same size")

        convs = []
        prev_feat = infeats
        for feature_size, kernel_size, stride in zip(
            feature_sizes, kernel_sizes, strides
        ):
            convs.append(_conv_bn_lrelu(prev_feat, feature_size, kernel_size, stride))
            prev_feat = feature_size
        # self.convs = nn.Sequential(*convs)
        self.convs = nn.ModuleList(convs)
        self.strides = strides

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        for conv, stride in zip(self.convs, self.strides):
            og_shape = x.shape[2:]
            x = conv(x)
            padding = _compute_pad(og_shape, x.shape[2:], stride)
            x = F.pad(x, padding)
        return x


class FlowNetCorr(nn.Module):
    """
    Implements FlowNetCorr
    """

    def __init__(
        self,
        correlation_patch_size: int = 3,
        infeats: int = 1,
        feature_extractor_feature_sizes: List[int] = [8, 32, 64],
        feature_extractor_kernel_sizes: List[int] = [7, 5, 5],
        feature_extractor_strides: List[int] = [2, 1, 1],
        redir_feats: int = 32,
    ):
        super().__init__()
        self.correlation_patch_size = correlation_patch_size
        self.feature_extractor = FeatureExtractor(  # (*fex_args, **fex_kwargs)
            infeats=infeats,
            feature_sizes=feature_extractor_feature_sizes,
            kernel_sizes=feature_extractor_kernel_sizes,
            strides=feature_extractor_strides,
        )
        self.conv_redir = _conv_bn_lrelu(
            infeats=feature_extractor_feature_sizes[-1],
            outfeats=redir_feats,
            kernel_size=1,
            stride=1,
        )
        # NOTE: this is not strictly like flownet because we just pop in a unet
        corr_out_feat = (1 + (correlation_patch_size * 2)) ** 3 + redir_feats
        self.unet = Unet3D(infeats=corr_out_feat)
        self.flow = nn.Conv3d(
            default_unet_features()[-1][-1], 3, kernel_size=3, padding=1
        )

        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        resize_factor = 1 / np.prod(feature_extractor_strides).item()
        self.flow_resize = ResizeTransform(resize_factor, ndims=3)
        self.refine = FlownetRefinementModule()

    def forward(self, moving_image: torch.Tensor, fixed_image: torch.Tensor, resize: bool=True):
        feat_mov = self.feature_extractor(moving_image)
        feat_fix = self.feature_extractor(fixed_image)
        corr = correlate_grad(
            feat_fix,
            feat_mov,
            self.correlation_patch_size,
            grid_sp=1,
            image_shape=feat_mov.shape[2:],
        ).unsqueeze(0)
        redir = self.conv_redir(feat_mov)
        unet_in = torch.cat([corr, redir], dim=1)
        unet_out = self.unet(unet_in)
        flow = self.flow(unet_out)
        if resize:
            flow = self.flow_resize(flow)
            flow = self.refine(flow)

        return flow


class FlownetRefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=3, out_channels=3, kernel_size=3, padding="same"
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = "linear"
        if ndims == 2:
            self.mode = "bi" + self.mode
        elif ndims == 3:
            self.mode = "tri" + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )

        # don't do anything if resize is 1
        return x


class Unet3D(nn.Module):
    """
    Modifed Unet from Voxelmoprh
    """

    def __init__(
        self,
        infeats=None,
        outfeats=None,
        nb_features=None,
        nb_levels=None,
        max_pool=2,
        feat_mult=1,
        nb_conv_per_level=1,
        half_res=False,
    ):
        """
        Parameters:
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        ndims = 3
        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]
        self.maxpool = max_pool

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for _, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

        self.outconv = None
        if outfeats != None:
            self.outconv = ConvBlock(ndims, self.final_nf, outfeats)

    def __pad_for_pooling(self, pre_pad: torch.Tensor, pooling_amount: int):
        padding = [0] * 2 * len(pre_pad.shape[2:])
        for i, sz in enumerate(pre_pad.shape[2:]):
            if sz % pooling_amount != 0:
                diff = sz - (pooling_amount * (sz // pooling_amount))
                half = diff // 2
                rem = diff - half
                padding[2 * i] = half
                padding[2 * i + 1] = rem
        return padding[::-1]

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        ndim = len(x.shape[2:])
        pad_history = [[0] * 2 * ndim]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            pad_history.append(self.__pad_for_pooling(x, self.maxpool[level]))
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = F.pad(x, pad_history.pop())
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        if self.outconv is not None:
            x = self.outconv(x)

        return x


class VxmDense3D(nn.Module):
    """
    Slightly modified version of VxmDense from Voxelmorph
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        nb_unet_conv_per_level=1,
        src_feats=1,
        trg_feats=1,
    ):
        """ 
        Parameters:
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            bidir: Enable bidirectional cost function. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = 3
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet3D(
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, source, target):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        preint_flow = pos_flow

        return preint_flow


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer: taken directly from voxelmorph
    """

    def __init__(self, size):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.float()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow, mode="bilinear"):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=mode)


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class FeatureExtractorVxm(nn.Module):
    """
    Slightly modified version of Voxelmorph's UNet
    """

    def __init__(
        self,
        infeats: int,
        outfeats: int = 12,
        nb_features: List[List[int]] = default_unet_features(),
        max_pool: Union[int, List[int]] = 2,
        nb_conv_per_level: int = 1,
        half_res: bool = False,
    ):
        super().__init__()

        # ensure correct dimensionality
        ndims = 3

        # cache some parameters
        self.half_res = half_res
        self.outfeats = outfeats

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        self.outconv = ConvBlock(ndims, prev_nf, outfeats)

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            # padding = [0]*2*len(x.shape)
            # for i, s in enumerate(x.shape[2:],start=2):
            #     if s%2 == 1:
            #         padding[2*i] = 1
            # padding = padding[::-1]
            # paddings.append(padding)
            # x = F.pad(x, padding)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        x = self.outconv(x)

        return x


class Cascade(nn.Module):
    def __init__(self, N: int = 2, similarity_function: str = "mi"):
        super().__init__()
        self.cascades = nn.ModuleList([Unet3D(7, 3) for _ in range(N)])
        self.similarity_function = mind_mse

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        flow: torch.Tensor,
        transformer: SpatialTransformer,
    ):
        for network in self.cascades:
            moved = transformer(moving, flow)
            similarity = self.similarity_function(moved, fixed)
            net_in = torch.concat((fixed, moving, flow, moved, similarity), dim=1)
            flow = flow + network(net_in)

        return flow
