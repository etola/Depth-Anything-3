from PIL import Image
import os
import numpy as np
import cv2

def crop_image_center(img: np.ndarray, target_size: int) -> tuple:
    # Get original dimensions
    org_height, org_width = img.shape[:2]

    if min(org_width, org_height) < target_size:
        raise ValueError(f"Image size {org_width}x{org_height} is smaller than target size {target_size}")

    center_x = org_width // 2
    center_y = org_height // 2
    half_size = target_size // 2

    left = max(0, center_x - half_size)
    right = min(org_width, center_x + half_size)

    upper = max(0, center_y - half_size)
    lower = min(org_height, center_y + half_size)


    img_cropped = img[upper:lower, left:right, :]
    original_coords = np.array([0, 0, target_size, target_size, org_width, org_height])

    assert img_cropped.shape[0] == target_size and img_cropped.shape[1] == target_size, \
        f"Cropped image size {img_cropped.shape[1]}x{img_cropped.shape[0]} does not match target size {target_size}x{target_size}"

    return img_cropped, original_coords


def load_and_preprocess_image_scale_then_crop(image_path: str, img_load_size: int, target_size: int) -> tuple:
    """
    Load and preprocess images by scaling to img_load_size and then cropping to target_size.
    Also returns the position information of original pixels after transformation. During scaling phase,
    we preserve the aspect ratio of the original image.

    Args:
        image_path (str): Path to image file
        img_load_size (int): Size to scale the image to after loading.
        target_size (int): Target size for both width and height.
    Returns:
        tuple: (
            np.ndarray: preprocessed image with shape (target_size, target_size, 3),
            np.ndarray: Array of shape (5) containing [x1, y1, x2, y2, width, height] for the image
        )
    Raises:
        ValueError: If the image path does not exist
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    if img_load_size < target_size:
        raise ValueError(f"Scaling size {img_load_size} is smaller than target size {target_size}")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    org_width, org_height = img.size

    max_dim = max(org_width, org_height)

    if img_load_size < target_size:
        scaled_image = np.array(img)
    else:
        new_w = int(org_width *  (img_load_size / max_dim))
        new_h = int(org_height * (img_load_size / max_dim))
        scaled_image = np.array(img.resize((new_w, new_h), Image.Resampling.BICUBIC))

    img_cropped, _cropped_coords = crop_image_center(scaled_image, target_size)
    original_coords = np.array([0, 0, target_size, target_size, org_width, org_height])

    return img_cropped, original_coords



def load_and_preprocess_image_center_crop(image_path: str, target_size: int) -> tuple:
    """
    Load and preprocess images by center crop to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path (str): Path to image file
        target_size (int): Target size for both width and height.
    Returns:
        tuple: (
            np.ndarray: preprocessed image with shape (target_size, target_size, 3),
            np.ndarray: Array of shape (5) containing [x1, y1, x2, y2, width, height] for the image
        )
    Raises:
        ValueError: If the image path does not exist
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    return crop_image_center(np.array(img), target_size)


# vggt-like square image loading with padding
def load_and_preprocess_image_square_padding(image_path: str, target_size: int) -> tuple:
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path (str): Path to image file
        target_size (int): Target size for both width and height.

    Returns:
        tuple: (
            np.ndarray: preprocessed image with shape (target_size, target_size, 3),
            np.ndarray: Array of shape (5) containing [x1, y1, x2, y2, width, height] for the image
        )

    Raises:
        ValueError: If the image path does not exist
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    org_width, org_height = img.size

    # Make the image square by padding the shorter dimension
    max_dim = max(org_width, org_height)

    # Calculate padding
    left = (max_dim - org_width) // 2
    top = (max_dim - org_height) // 2

    # Calculate scale factor for resizing
    scale = target_size / max_dim

    # Calculate final coordinates of original image in target space
    x1 = left * scale
    y1 = top * scale
    x2 = (left + org_width) * scale
    y2 = (top + org_height) * scale

    # Store original image coordinates and scale
    original_coords = np.array([x1, y1, x2, y2, org_width, org_height])

    # Create a new black square image and paste original
    square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square_img.paste(img, (left, top))

    # Resize to target size
    square_img = np.array(square_img.resize((target_size, target_size), Image.Resampling.BICUBIC))

    return square_img, original_coords


# vggt-like image loading with center crop to target size
def load_and_preprocess_image_crop(image_path: str, target_size: int) -> tuple:
    """
    Load and preprocess images by center crop to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path (str): Path to image file
        target_size (int): Target size for both width and height.

    Returns:
        tuple: (
            np.ndarray: preprocessed image with shape (target_size, target_size, 3),
            np.ndarray: Array of shape (5) containing [x1, y1, x2, y2, width, height] for the image
        )

    Raises:
        ValueError: If the image path does not exist
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    org_width, org_height = img.size

    # Width normalization with proportional height adjustment
    scaled_width = target_size
    scaled_height = round(org_height * (scaled_width / org_width) / 14) * 14

    img = np.array(img.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC))

    if scaled_height > target_size:
        start_y = (scaled_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]

    h, w = img.shape[:2]
    original_coords = np.array([0, 0, w, h, org_width, org_height])

    return img, original_coords


# vggt-like square image loading - destroys aspect ratio
def load_and_preprocess_image_square(image_path: str, target_size: int) -> tuple:
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path (str): Path to image file
        target_size (int): Target size for both width and height.

    Returns:
        tuple: (
            np.ndarray: preprocessed image with shape (target_size, target_size, 3),
            np.ndarray: Array of shape (5) containing [x1, y1, x2, y2, width, height] for the image
        )

    Raises:
        ValueError: If the image path does not exist
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    org_width, org_height = img.size

    square_img = np.array(img.resize((target_size, target_size), Image.Resampling.BICUBIC))

    # Store original image coordinates and scale
    original_coords = np.array([0, 0, target_size, target_size, org_width, org_height])

    return square_img, original_coords



# scale images preserving aspect ratio where max dimension is target_size
def load_and_preprocess_image_keep_aspect(image_path: str, target_size: int) -> tuple:
    """
    Load and preprocess images by resizing to target size while preserving aspect ratio.

    Args:
        image_path (str): Path to image file
        target_size (int): Target size for both width and height.
    Returns:
        np.ndarray: preprocessed image with shape (target_size, target_size, 3),
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    org_width, org_height = img.size

    scale = target_size / max(org_width, org_height)

    new_width = int(org_width * scale)
    new_height = int(org_height * scale)

    scaled_img = np.array(img.resize((new_width, new_height), Image.Resampling.BICUBIC))

    original_coords = np.array([0, 0, new_width, new_height, org_width, org_height])

    return scaled_img, original_coords


def load_and_preprocess_image_keep_aspect_14_multiple(image_path: str, target_size: int) -> tuple:

    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    # Open image
    img = Image.open(image_path)

    # If there's an alpha channel, blend onto white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    img = img.convert("RGB")

    # Get original dimensions
    org_width, org_height = img.size

    if org_width > org_height:
        resized_w = target_size
        resized_h = round(org_height * (resized_w / org_width) / 14) * 14
    else:
        resized_h = target_size
        resized_w = round(org_width * (resized_h / org_height) / 14) * 14

    image_resized = np.array( img.resize((resized_w, resized_h), Image.Resampling.BICUBIC) )
    original_coords = np.array([0, 0, resized_w, resized_h, org_width, org_height])

    return image_resized, original_coords


def load_and_preprocess_image(image_path: str, target_w: int, target_h: int) -> tuple:
    """
    Load and preprocess images by resizing to target size.
    Also returns the position information of original pixels after transformation.
    Args:
        image_path (str): Path to image file
        target_w (int): Target width
        target_h (int): Target height

    assumes target_w/target_h is the aspect ratio of the original image. throws error if not.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    org_width, org_height = img.size

    scaled_img = np.array(img.resize((target_w, target_h), Image.Resampling.BICUBIC))
    original_coords = np.array([0, 0, target_w, target_h, org_width, org_height])
    return scaled_img, original_coords

def load_and_resize_keep_aspect(image_path: str, target_w: int, target_h: int) -> tuple:
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    ori_w, ori_h = img.size

    ori_area = ori_h * ori_w
    tar_area = target_h * target_w
    scale = scale = (tar_area / ori_area) ** 0.5
    resize_h = ori_h * scale
    resize_w = ori_w * scale
    resize_h = max(16, int(round(resize_h / 16)) * 16)
    resize_w = max(16, int(round(resize_w / 16)) * 16)

    original_coords = np.array([0, 0, resize_w, resize_h, ori_w, ori_h])

    if ori_h == resize_h and ori_w == resize_w:
        return img, original_coords

    if scale < 1:
        scaled_img = np.array(img.resize((resize_w, resize_h), Image.Resampling.BOX))
    else:
        scaled_img = np.array(img.resize((resize_w, resize_h), Image.Resampling.BICUBIC))

    return scaled_img, original_coords
