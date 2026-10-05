"""Spectral core for VALLIS-3C 1.6.0."""

from .fas_spectral import FASSpectralModel
from .site_texture import SiteTextureModel

__version__ = "1.6.0"

__all__ = [
    "FASSpectralModel",
    "SiteTextureModel",
]
