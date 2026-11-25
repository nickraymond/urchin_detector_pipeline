"""One helper for every pixel read of a frame (DESIGN D27): applies the EXIF orientation, so
SAM, the masks and the pages all see the same upright image. Still images stay byte-for-byte
copies on disk (D21); only what is read from them is turned upright."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def open_upright(path: Path) -> Image.Image:
    """The decoded image, turned upright by its EXIF orientation (no-op without one)."""
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im)


def upright(im: Image.Image) -> Image.Image:
    """The same for an image that is already open."""
    return ImageOps.exif_transpose(im)
