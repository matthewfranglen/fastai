from ...core import *
from pathlib import Path
from torch import nn
from torch.utils import model_zoo

__all__ = ['efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4','efficientnetb5','efficientnetb6','efficientnetb7']


def _efficientnet(arch: str, blocks: List[EfficientNetBlockParameters], pretrained: bool, progress: bool):
    params = _MODEL_PARAMS[arch]
    model = EfficientNet(blocks, params)
    if pretrained:
        mapping = json.reads((Path(__file__).parent/'_efficientnet-mapping.json').read_text())[arch]
        state_dict = model_zoo.load_url(arch.url, progress=progress)
        state_dict = {mapping[k]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model

def efficientnetb0(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b0'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb1(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b1'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb2(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b2'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb3(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b3'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb4(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b4'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb5(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b5'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb6(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b6'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)

def efficientnetb7(pretrained=False, progress=True, **kwargs):
    return _efficientnet(arch=_MODEL_PARAMS['efficientnet-b7'], blocks=_BLOCK_PARAMS, pretrained=pretrained, progress=progress)


Value2d = Union[int, Tuple[int, int]]

class EfficientNet(nn.Sequential):
    """ https://arxiv.org/pdf/1905.11946.pdf """

    def __init__(self, blocks: List[EfficientNetBlockParameters], model_params: EfficientNetParameters):
        pcn = partial(
            pad_conv_bn,
            img_size=model_params.img_size,
            bn_momentum=1 - model_params.bn_momentum,
            bn_epsilon=model_params.bn_epsilon,
        )

        in_channels = 3
        out_channels = round_filters(32, model_params)

        layers = []
        layers.append(
            pcn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False)
        )

        block_layers = []
        block_count = sum(max(1, block.num_repeat) for block in blocks)
        for block in blocks:
            # Update block input and output filters based on depth multiplier.
            block = block.clone(
                input_filters=round_filters(block.input_filters, model_params),
                output_filters=round_filters(block.output_filters, model_params),
                num_repeat=round_repeats(block.num_repeat, model_params),
            )

            # The first block needs to take care of stride and filter size increase.
            block_layers.append(mbconv(block, model_params))
            if block.num_repeat > 1:
                block = block.clone(input_filters=block.output_filters, stride=1)
            for _ in range(block.num_repeat - 1):
                dropout = model_params.drop_connect_rate * len(block_layers) / block_count if dropout else 0
                block_layers.append(mbconv(block, model_params, dropout=dropout))
        layers.append(nn.Sequential(*block_layers))

        in_channels = block.output_filters
        out_channels = round_filters(1280, model_params)
        layers.append(
            pcn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        )
        layers.append(EfficientNetHead(out_channels, model_params))

        super().__init__(*layers)


def mbconv(block_parameters: EfficientNetBlockParameters, model_params: EfficientNetParameters, dropout: float = 0) -> nn.Module:
    """ Mobile Inverted Bottleneck with Squeeze and Excitation """
    layers = []
    output_channels = block_parameters.input_filters * block_parameters.expand_ratio

    pcn = partial(
        pad_conv_bn,
        img_size=model_params.img_size,
        bn_momentum=1 - model_params.bn_momentum,
        bn_epsilon=model_params.bn_epsilon,
    )

    if block_parameters.expand_ratio != 1:
        layers.append(
            pcn(in_channels=block_parameters.input_filters, out_channels=output_channels, kernel_size=1, bias=False)
        )

    layers.append(
        pcn(
            in_channels=output_channels,
            out_channels=output_channels,
            groups=output_channels,
            kernel_size=block_parameters.kernel_size,
            stride=block_parameters.stride,
            bias=False,
        )
    )

    if block_parameters.se_ratio is not None and 0 < block_parameters.se_ratio <= 1:
        layers.append(
            SqueezeAndExcitation(block_parameters=block_parameters, output_channels=output_channels)
        )

    layers.append(
        pcn(in_channels=output_channels, out_channels=block_parameters.output_filters, kernel_size=1, bias=False, has_swish=False)
    )

    if block_parameters.id_skip and block_parameters.stride == 1 and block_parameters.input_filters == block_parameters.output_filters:
        if dropout:
            return SkipDropConnection(dropout, *layers)
        return SkipConnection(*layers)
    return nn.Sequential(*layers)


class SqueezeAndExcitation(nn.Sequential):
    def __init__(
        self, block_parameters: EfficientNetBlockParameters, output_channels: int
    ):
        num_squeezed_channels = max(
            1, int(block_parameters.input_filters * block_parameters.se_ratio)
        )
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            ),
            MemoryEfficientSwish(),
            nn.Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
        )

    def forward(self, _in, *others):
        squeezed = super().forward(_in, *others)
        return torch.sigmoid(squeezed) * _in


class SkipConnection(nn.Sequential):
    def forward(self, _in, *others):
        out = super().forward(_in, *others)
        return _in + out


class SkipDropConnection(nn.Sequential):
    def __init__(self, dropout: float, *layers):
        super().__init__(*layers)
        self.dropout = dropout

    def forward(self, _in, *others):
        out = super().forward(_in, *others)
        return F.dropout(_in, p=self.dropout, training=self.training) + out


def pad(conv: nn.Conv2d, img_size: Value2d) -> Optional[nn.Module]:
    stride_height, stride_width = listify(conv.stride, 2)
    dilation_height, dilation_width = listify(conv.dilation, 2)
    kernel_height, kernel_width = conv.weight.shape[-2:]
    image_height, image_width = listify(img_size, 2)

    output_height = math.ceil(image_height / stride_height)
    output_width = math.ceil(image_width / stride_width)

    pad_h = _calculate_padding(
        _in=image_height,
        out=output_height,
        stride=stride_height,
        dilation=dilation_height,
        kernel=kernel_height,
    )
    pad_w = _calculate_padding(
        _in=image_width,
        out=output_width,
        stride=stride_width,
        dilation=dilation_width,
        kernel=kernel_width,
    )

    if pad_h > 0 or pad_w > 0:
        return nn.ZeroPad2d(
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        )
    return None


def _calculate_padding(_in: int, out: int, stride: int, dilation: int, kernel: int) -> int:
    return max((out - 1) * stride + (kernel - 1) * dilation + 1 - _in, 0)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _in, *_):
        result = _in * torch.sigmoid(_in)
        ctx.save_for_backward(_in)
        return result

    @staticmethod
    def backward(ctx, gradient_output):
        _in = ctx.saved_variables[0]
        sigmoid_in = torch.sigmoid(_in)
        return gradient_output * (sigmoid_in * (1 + _in * (1 - sigmoid_in)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, _in, *_):
        return SwishImplementation.apply(_in)


def pad_conv_bn(
    *,
    img_size: Value2d,
    bn_momentum: float,
    bn_epsilon: float,
    in_channels: int,
    out_channels: int,
    kernel_size: Value2d,
    stride: Value2d = 1,
    padding: Value2d = 0,
    dilation: Value2d = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    has_swish: bool = True,
) -> nn.Module:
    """ A repeated pattern in efficientnet is pad -> conv2d -> batch norm -> swish.
        This wraps that in a sequential block. """
    layers = []
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
    padding = pad(conv, img_size)
    if padding:
        layers.append(padding)
    layers.append(conv)
    layers.append(
        nn.BatchNorm2d(
            num_features=out_channels,
            momentum=bn_momentum,
            eps=bn_epsilon,
        )
    )
    if has_swish:
        layers.append(MemoryEfficientSwish())

    return nn.Sequential(*layers)

class EfficientNetHead(nn.Module):
    def __init__(self, in_channels: int, model_params: EfficientNetParameters):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(model_params.dropout)
        self.fc = nn.Linear(
            in_channels, model_params.num_classes
        )

    def forward(self, _in):
        batch_size = _in.shape[0]
        out = self.avg_pool(_in)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        return self.fc(out)


def round_filters(filters: int, model_params: EfficientNetParameters) -> int:
    """ Calculate and round number of filters based on depth multiplier. """
    if not model_params.width_coefficient:
        return filters
    divisor = model_params.depth_divisor
    min_depth = model_params.min_depth
    filters *= model_params.width_coefficient
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, model_params: EfficientNetParameters) -> int:
    """ Round number of filters based on depth multiplier. """
    if not model_params.depth_coefficient:
        return repeats
    return int(math.ceil(model_params.depth_coefficient * repeats))

@dataclass(frozen=True)
class EfficientNetParameters:
    url: str

    width_coefficient: float
    depth_coefficient: float
    img_size: int
    dropout: float

    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    drop_connect_rate: float = 0.2
    num_classes: int = 1_000
    depth_divisor: int = 8
    min_depth: float = None

    def clone(self, **params) -> EfficientNetParameters:
        """ Copy and change these parameters """
        parameters = {**asdict(self), **params}
        return EfficientNetParameters(**parameters)


@dataclass(frozen=True)
class EfficientNetBlockParameters:
    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    id_skip: bool
    stride: List[int]
    se_ratio: float

    def clone(self, **params) -> EfficientNetParameters:
        """ Copy and change these parameters """
        parameters = {**asdict(self), **params}
        return EfficientNetBlockParameters(**parameters)


_MODEL_PARAMS = {
    "efficientnet-b0": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth",
        width_coefficient=1.0,
        depth_coefficient=1.0,
        img_size=224,
        dropout=0.2,
    ),
    "efficientnet-b1": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth",
        width_coefficient=1.0,
        depth_coefficient=1.1,
        img_size=240,
        dropout=0.2,
    ),
    "efficientnet-b2": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth",
        width_coefficient=1.1,
        depth_coefficient=1.2,
        img_size=260,
        dropout=0.3,
    ),
    "efficientnet-b3": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth",
        width_coefficient=1.2,
        depth_coefficient=1.4,
        img_size=300,
        dropout=0.3,
    ),
    "efficientnet-b4": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth",
        width_coefficient=1.4,
        depth_coefficient=1.8,
        img_size=380,
        dropout=0.4,
    ),
    "efficientnet-b5": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth",
        width_coefficient=1.6,
        depth_coefficient=2.2,
        img_size=456,
        dropout=0.4,
    ),
    "efficientnet-b6": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth",
        width_coefficient=1.8,
        depth_coefficient=2.6,
        img_size=528,
        dropout=0.5,
    ),
    "efficientnet-b7": EfficientNetParameters(
        url="http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth",
        width_coefficient=2.0,
        depth_coefficient=3.1,
        img_size=600,
        dropout=0.5,
    ),
}

_BLOCK_PARAMS = [
    EfficientNetBlockParameters(
        kernel_size=3,
        num_repeat=1,
        input_filters=32,
        output_filters=16,
        expand_ratio=1,
        id_skip=True,
        stride=[1],
        se_ratio=0.25,
    ),
    EfficientNetBlockParameters(
        kernel_size=3,
        num_repeat=2,
        input_filters=16,
        output_filters=24,
        expand_ratio=6,
        id_skip=True,
        stride=[2],
        se_ratio=0.25,
    ),
    EfficientNetBlockParameters(
        kernel_size=5,
        num_repeat=2,
        input_filters=24,
        output_filters=40,
        expand_ratio=6,
        id_skip=True,
        stride=[2],
        se_ratio=0.25,
    ),
    EfficientNetBlockParameters(
        kernel_size=3,
        num_repeat=3,
        input_filters=40,
        output_filters=80,
        expand_ratio=6,
        id_skip=True,
        stride=[2],
        se_ratio=0.25,
    ),
    EfficientNetBlockParameters(
        kernel_size=5,
        num_repeat=3,
        input_filters=80,
        output_filters=112,
        expand_ratio=6,
        id_skip=True,
        stride=[1],
        se_ratio=0.25,
    ),
    EfficientNetBlockParameters(
        kernel_size=5,
        num_repeat=4,
        input_filters=112,
        output_filters=192,
        expand_ratio=6,
        id_skip=True,
        stride=[2],
        se_ratio=0.25,
    ),
    EfficientNetBlockParameters(
        kernel_size=3,
        num_repeat=1,
        input_filters=192,
        output_filters=320,
        expand_ratio=6,
        id_skip=True,
        stride=[1],
        se_ratio=0.25,
    ),
]
