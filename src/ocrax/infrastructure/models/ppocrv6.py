"""Flax NNX PP-OCRv6 medium recognizer (CTC inference/training path).

Architecture follows PaddleOCR's rec_lcnetv4.py, rnn.py, rec_svtrnet.py,
rec_multi_head.py, and PP-OCRv6_medium_rec.yml. Paddle weights are not loaded
implicitly; use the checkpoint conversion utilities and pass parity first.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


REC_CONFIG = {
    "tiny": {
        "stem": (24, 48), "stem_type": "simple",
        "blocks2": [(3, 48, 48, 1, True)],
        "blocks3": [(3, 48, 48, 1, False)],
        "blocks4": [(3, 48, 96, (2, 1), False), (3, 96, 96, 1, True), (3, 96, 96, 1, False)],
        "blocks5": [(3, 96, 160, (2, 1), False), (3, 160, 160, 1, True), (3, 160, 160, 1, False), (3, 160, 160, 1, False)],
        "blocks6": [],
    },
    "small": {
        "stem": (48, 96), "stem_type": "branch",
        "blocks2": [(3, 96, 96, 1, True)],
        "blocks3": [(3, 96, 96, 1, False), (3, 96, 96, 1, False)],
        "blocks4": [(3, 96, 192, (2, 1), False), (3, 192, 192, 1, True), (3, 192, 192, 1, False), (3, 192, 192, 1, True), (3, 192, 192, 1, False), (3, 192, 192, 1, True), (3, 192, 192, 1, False)],
        "blocks5": [(3, 192, 384, (2, 1), False), (3, 384, 384, 1, True), (3, 384, 384, 1, False)],
        "blocks6": [],
    },
    "medium": {
        "stem": (64, 128), "stem_type": "branch",
        "blocks2": [(3, 128, 128, 1, True)],
        "blocks3": [(3, 128, 256, 1, False), (3, 256, 256, 1, False), (3, 256, 256, 1, True)],
        "blocks4": [(3, 256, 512, (2, 1), False), (3, 512, 512, 1, True), (3, 512, 512, 1, False), (3, 512, 512, 1, True), (3, 512, 512, 1, False), (3, 512, 512, 1, True), (3, 512, 512, 1, False)],
        "blocks5": [(3, 512, 768, (2, 1), False), (3, 768, 768, 1, True), (3, 768, 768, 1, False)],
        "blocks6": [],
    },
}


def _stride_pair(stride: int | Sequence[int]) -> tuple[int, int]:
    return (stride, stride) if isinstance(stride, int) else tuple(stride)


def _explicit_padding(kernel: int | Sequence[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    kh, kw = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
    return ((kh // 2, kh // 2), (kw // 2, kw // 2))


def _kernel_pair(kernel: int | Sequence[int]) -> tuple[int, int]:
    return (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)


class ConvBN(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int | Sequence[int], *, stride: int | Sequence[int] = 1, groups: int = 1, padding: str | tuple = "explicit", use_bias: bool = True, rngs: nnx.Rngs, bn_scale_init=1.0):
        self.conv = nnx.Conv(
            in_ch, out_ch, _kernel_pair(kernel), strides=_stride_pair(stride),
            padding="SAME" if padding == "same" else _explicit_padding(kernel),
            feature_group_count=groups, use_bias=use_bias, rngs=rngs,
        )
        self.bn = nnx.BatchNorm(
            out_ch, use_running_average=False, axis=-1, momentum=0.9,
            epsilon=1e-5,
            scale_init=nnx.initializers.constant(bn_scale_init), rngs=rngs,
        )

    def __call__(self, x, *, training: bool, activation: str | None = None):
        x = self.conv(x)
        # The converted inference checkpoint folds source BatchNorm into the
        # convolution and represents it as an identity BN. Keep running
        # statistics frozen during fine-tuning to preserve that initialization.
        x = self.bn(x, use_running_average=True)
        if activation == "relu":
            return jax.nn.relu(x)
        if activation == "gelu":
            return jax.nn.gelu(x, approximate=False)
        if activation == "silu":
            return jax.nn.silu(x)
        if activation == "hardswish":
            return x * jax.nn.relu6(x + 3) / 6
        return x


class ConvBNAct(nnx.Module):
    def __init__(self, in_ch, out_ch, kernel, *, stride=1, groups=1, padding="explicit", use_act=True, rngs):
        self.use_act = use_act
        self.block = ConvBN(in_ch, out_ch, kernel, stride=stride, groups=groups, padding=padding, rngs=rngs)

    def __call__(self, x, *, training: bool):
        return self.block(x, training=training, activation="relu" if self.use_act else None)


class StemBlock(nnx.Module):
    def __init__(self, mid_ch: int, out_ch: int, rngs: nnx.Rngs):
        self.stem1 = ConvBNAct(3, mid_ch, 3, stride=2, rngs=rngs)
        self.stem2a = ConvBNAct(mid_ch, mid_ch // 2, 2, padding="same", rngs=rngs)
        self.stem2b = ConvBNAct(mid_ch // 2, mid_ch, 2, padding="same", rngs=rngs)
        self.stem3 = ConvBNAct(mid_ch * 2, mid_ch, 3, stride=2, rngs=rngs)
        self.stem4 = ConvBNAct(mid_ch, out_ch, 1, rngs=rngs)

    def __call__(self, x, *, training: bool):
        x = self.stem1(x, training=training)
        x2 = self.stem2b(self.stem2a(x, training=training), training=training)
        # Paddle MaxPool2D(kernel=2, stride=1, padding="SAME"). Explicit padding
        # keeps output dimensions stable; parity validation checks border alignment.
        x1 = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 1, 1, 1),
            ((0, 0), (0, 1), (0, 1), (0, 0)),
        )
        return self.stem4(self.stem3(jnp.concatenate([x1, x2], axis=-1), training=training), training=training)


class SqueezeExcite(nnx.Module):
    def __init__(self, channels: int, rngs: nnx.Rngs):
        hidden = channels // 4
        self.fc1 = nnx.Conv(channels, hidden, (1, 1), use_bias=True, rngs=rngs)
        self.fc2 = nnx.Conv(hidden, channels, (1, 1), use_bias=True, rngs=rngs)

    def __call__(self, x):
        scale = jnp.mean(x, axis=(1, 2), keepdims=True)
        scale = jax.nn.relu(self.fc1(scale))
        scale = self.fc2(scale)
        scale = jnp.clip((scale + 3.0) / 6.0, 0.0, 1.0)
        return x * scale


class RepDWConv(nnx.Module):
    """Unfused train-time 3x3 depthwise + 1x1 depthwise + identity/BN."""
    def __init__(self, channels: int, rngs: nnx.Rngs):
        self.branch3 = ConvBN(channels, channels, 3, groups=channels, rngs=rngs)
        self.branch1 = nnx.Conv(channels, channels, (1, 1), feature_group_count=channels, use_bias=False, rngs=rngs)
        self.identity_bn = nnx.BatchNorm(channels, use_running_average=False, axis=-1, momentum=0.9, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, *, training: bool):
        y = self.branch3(x, training=training) + self.branch1(x) + x
        return self.identity_bn(y, use_running_average=True)


class LCNetV4Block(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int | Sequence[int], kernel: int, use_se: bool, rngs: nnx.Rngs):
        self.stride = stride
        self.residual = isinstance(stride, int) and stride == 1 and in_ch == out_ch
        self.rep_dw = self.residual
        self.dw = RepDWConv(in_ch, rngs) if self.rep_dw else ConvBN(in_ch, in_ch, kernel, stride=stride, groups=in_ch, rngs=rngs)
        self.se = SqueezeExcite(in_ch, rngs) if use_se else None
        hidden = in_ch * 2
        self.expand = ConvBN(in_ch, hidden, 1, rngs=rngs)
        # Paddle initializes the final BN scale to zero for residual blocks.
        self.compress = ConvBN(hidden, out_ch, 1, rngs=rngs, bn_scale_init=0.0 if self.residual else 1.0)

    def __call__(self, x, *, training: bool):
        y = self.dw(x, training=training) if self.rep_dw else self.dw(x, training=training)
        if self.se is not None:
            y = self.se(y)
        z = self.expand(y, training=training, activation="gelu")
        z = self.compress(z, training=training)
        return y + z if self.residual else z


class PPLCNetV4Rec(nnx.Module):
    def __init__(self, model_size: str = "medium", rngs: nnx.Rngs | None = None):
        if model_size not in REC_CONFIG:
            raise ValueError(f"Unknown PP-OCRv6 recognition size {model_size!r}")
        rngs = rngs or nnx.Rngs(0)
        cfg = REC_CONFIG[model_size]
        mid, out = cfg["stem"]
        if cfg["stem_type"] == "branch":
            self.conv1 = StemBlock(mid, out, rngs)
        else:
            self.conv1a = ConvBN(3, mid, 3, stride=2, rngs=rngs)
            self.conv1b = ConvBN(mid, out, 3, stride=2, rngs=rngs)
        for stage in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            setattr(self, stage, nnx.List([LCNetV4Block(inc, outc, stride, kernel, se, rngs) for kernel, inc, outc, stride, se in cfg[stage]]))
        self.model_size = model_size

    def __call__(self, x, *, training: bool):
        if hasattr(self, "conv1"):
            x = self.conv1(x, training=training)
        else:
            x = jax.nn.gelu(self.conv1a(x, training=training), approximate=False)
            x = jax.nn.gelu(self.conv1b(x, training=training), approximate=False)
        for stage in (self.blocks2, self.blocks3, self.blocks4, self.blocks5, self.blocks6):
            for block in stage:
                x = block(x, training=training)
        if training:
            return _adaptive_avg_pool(x, 1, 40)
        return _avg_pool_valid(x, 3, 2)


def _avg_pool_valid(x, kh: int, kw: int):
    b, h, w, c = x.shape
    sh, sw = kh, kw
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    x = x[:, :oh * sh, :ow * sw].reshape(b, oh, kh, ow, kw, c)
    return jnp.mean(x, axis=(2, 4))


def _adaptive_avg_pool(x, out_h: int, out_w: int):
    _, h, w, _ = x.shape
    rows = []
    for oy in range(out_h):
        y0, y1 = oy * h // out_h, ((oy + 1) * h + out_h - 1) // out_h
        cols = []
        for ox in range(out_w):
            x0, x1 = ox * w // out_w, ((ox + 1) * w + out_w - 1) // out_w
            cols.append(jnp.mean(x[:, y0:y1, x0:x1, :], axis=(1, 2), keepdims=True))
        rows.append(jnp.concatenate(cols, axis=2))
    return jnp.concatenate(rows, axis=1)


class ConvBNActSwish(nnx.Module):
    def __init__(self, in_ch, out_ch, kernel, *, groups=1, rngs):
        self.block = ConvBN(in_ch, out_ch, kernel, groups=groups, rngs=rngs, use_bias=True)

    def __call__(self, x, *, training: bool):
        return jax.nn.silu(self.block(x, training=training))


class AttentionBlock(nnx.Module):
    """SVTR global-attention block with Paddle's prenorm=False ordering."""
    def __init__(self, dims: int, depth_index: int, rngs: nnx.Rngs, mlp_ratio: float = 4.0, heads: int = 8, dropout_rate: float = 0.1):
        if dims % heads:
            raise ValueError(f"dims {dims} must divide evenly across {heads} heads")
        self.heads = heads
        self.head_dim = dims // heads
        self.norm1 = nnx.LayerNorm(dims, epsilon=1e-5, rngs=rngs)
        self.qkv = nnx.Linear(dims, 3 * dims, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(dims, dims, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dims, epsilon=1e-5, rngs=rngs)
        self.fc1 = nnx.Linear(dims, int(dims * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dims * mlp_ratio), dims, rngs=rngs)
        self.attn_dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.drop1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.drop2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.drop3 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.depth_index = depth_index

    def __call__(self, x, *, training: bool):
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, self.head_dim)
        q, k, v = (jnp.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3))
        attn = jnp.einsum("bhtd,bhsd->bhts", q * (self.head_dim ** -0.5), k)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        attn = self.attn_dropout(attn, deterministic=not training)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v).transpose(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.drop1(self.proj(y), deterministic=not training)
        y = jax.nn.silu(self.fc1(self.norm2(x)))
        y = self.drop2(y, deterministic=not training)
        y = self.fc2(y)
        return x + self.drop3(y, deterministic=not training)


class LightSVTR(nnx.Module):
    def __init__(self, in_ch: int = 768, dims: int = 192, depth: int = 2, local_kernel: int = 7, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(1)
        self.conv_reduce = ConvBNActSwish(in_ch, dims, 1, rngs=rngs)
        self.local = ConvBNActSwish(dims, dims, (1, local_kernel), groups=dims, rngs=rngs)
        self.blocks = nnx.List([AttentionBlock(dims, i, rngs) for i in range(depth)])
        self.norm = nnx.LayerNorm(dims, epsilon=1e-6, rngs=rngs)
        self.skip_conv = ConvBNActSwish(in_ch, dims, 1, rngs=rngs)

    def __call__(self, x, *, training: bool):
        # x is NHWC [B,1,40,768] after PPLCNetV4 and Im2Seq-equivalent pooling.
        skip = self.skip_conv(x, training=training)
        z = self.conv_reduce(x, training=training)
        z = z + self.local(z, training=training)
        b, h, w, c = z.shape
        z = z.reshape(b, h * w, c)
        for block in self.blocks:
            z = block(z, training=training)
        z = self.norm(z).reshape(b, h, w, c)
        return z + skip


class PP_OCRv6MediumRec(nnx.Module):
    """PP-OCRv6 medium recognition CTC path.

    The shipped inference graph has the CTC path only. The configured NRTR
    auxiliary training head is a separate port item and cannot be recovered
    from this inference graph unless a training checkpoint is supplied.
    """

    def __init__(self, num_classes: int = 18710, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.backbone = PPLCNetV4Rec("medium", rngs)
        self.neck = LightSVTR(768, 192, 2, 7, rngs)
        self.ctc = nnx.Linear(192, num_classes, rngs=rngs)

    def __call__(self, images_nchw, *, training: bool = False):
        x = jnp.transpose(images_nchw, (0, 2, 3, 1))
        x = self.backbone(x, training=training)
        x = self.neck(x, training=training)
        x = x[:, 0, :, :]
        logits = self.ctc(x)
        if training:
            return logits
        return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    def forward_intermediates(self, images_nchw) -> dict[str, jax.Array]:
        """Return named inference activations as NCHW/NLC arrays for parity diagnosis."""
        values = {}
        x = jnp.transpose(images_nchw, (0, 2, 3, 1))
        x = self.backbone.conv1(x, training=False)
        values["backbone_stem"] = jnp.transpose(x, (0, 3, 1, 2))
        for stage_name in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            for block in getattr(self.backbone, stage_name):
                x = block(x, training=False)
            if len(getattr(self.backbone, stage_name)):
                values[f"backbone_{stage_name}"] = jnp.transpose(x, (0, 3, 1, 2))
        x = _avg_pool_valid(x, 3, 2)
        values["backbone_pool"] = jnp.transpose(x, (0, 3, 1, 2))

        skip = self.neck.skip_conv(x, training=False)
        values["neck_skip"] = jnp.transpose(skip, (0, 3, 1, 2))
        z = self.neck.conv_reduce(x, training=False)
        values["neck_reduce"] = jnp.transpose(z, (0, 3, 1, 2))
        local = self.neck.local(z, training=False)
        values["neck_local"] = jnp.transpose(local, (0, 3, 1, 2))
        z = z + local
        values["neck_local_sum"] = jnp.transpose(z, (0, 3, 1, 2))
        b, h, w, c = z.shape
        z = z.reshape(b, h * w, c)
        for i, block in enumerate(self.neck.blocks):
            z = block(z, training=False)
            values[f"neck_block_{i}"] = z
        z = self.neck.norm(z)
        values["neck_norm"] = z
        z = z.reshape(b, h, w, c) + skip
        values["neck_out"] = jnp.transpose(z, (0, 3, 1, 2))
        sequence = z[:, 0, :, :]
        values["sequence"] = sequence
        logits = self.ctc(sequence)
        values["ctc_logits"] = logits
        values["probabilities"] = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        return values


def ctc_greedy_decode(probabilities, characters: Sequence[str], blank_id: int = 0):
    """Greedy CTC decode matching PaddleOCR's blank-first character layout."""
    ids = jnp.argmax(probabilities, axis=-1)
    outputs = []
    for row in ids.tolist():
        previous = None
        chars = []
        for token in row:
            if token != previous and token != blank_id:
                if token - 1 >= len(characters):
                    raise ValueError(f"Token ID {token} exceeds dictionary size {len(characters)}")
                chars.append(characters[token - 1])
            previous = token
        outputs.append("".join(chars))
    return outputs
