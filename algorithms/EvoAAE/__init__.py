"""
EvoAAE - 进化对抗自编码器用于工业物联网无监督异常检测
论文实现: Evolutionary Adversarial Autoencoder for Unsupervised Anomaly Detection of IIoT
"""

from .preprocessing import (
    spectral_residual_transform,
    sliding_window_sequence,
    preprocess_data_for_evoaae
)

from .models import (
    ConvEncoder,
    ConvDecoder,
    ConvDiscriminator,
    AdversarialAutoencoderWithDualDiscriminator
)

from .pso_optimizer import BinaryPSO
from .evoaae_model import EvoAAE

__version__ = "1.0.0"
__author__ = "论文实现"

__all__ = [
    'spectral_residual_transform',
    'sliding_window_sequence',
    'preprocess_data_for_evoaae',
    'ConvEncoder',
    'ConvDecoder',
    'ConvDiscriminator',
    'AdversarialAutoencoderWithDualDiscriminator',
    'BinaryPSO',
    'EvoAAE'
]