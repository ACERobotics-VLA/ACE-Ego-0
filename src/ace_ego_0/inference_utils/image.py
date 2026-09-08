"""Image helpers used by the ACE-Ego-0 inference path."""

from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """Resize nested PIL images while preserving their list structure."""
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_images(image, target_size) for image in images]
    raise TypeError(f"Expected a PIL image or list of images, got {type(images).__name__}.")
