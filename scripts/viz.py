import jax
import jax.numpy as jnp
from flax import nnx

from ocrax.infrastructure.models.ppocrv6 import LightSVTR, PPLCNetV4Rec

class PP_OCRv6MediumRec(nnx.Module):
    """PP-OCRv6 medium recognition inference graph in Flax NNX."""

    def __init__(self, num_classes: int = 18710, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.backbone = PPLCNetV4Rec("medium", rngs)
        self.neck = LightSVTR(in_channels=768, out_channels=192, depth=2, kernel_size=7, rngs=rngs)
        self.ctc = nnx.Linear(192, num_classes, rngs=rngs)

    def __call__(self, images_nchw: jax.Array, *, training: bool = False) -> jax.Array:
        # Transpose NCHW input to NHWC for standard JAX convolutions
        x = jnp.transpose(images_nchw, (0, 2, 3, 1))
        x = self.backbone(x, training=training)
        x = self.neck(x, training=training)
        # Squeeze spatial height to produce a 1D sequence across width (B, 40, 192)
        sequence = x[:, 0, :, :]
        logits = self.ctc(sequence)
        if training:
            return logits
        return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
