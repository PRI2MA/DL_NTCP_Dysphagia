import torch
import random
import numpy as np
from monai.transforms import (
    Compose,
    Rand3DElastic,
    RandAdjustContrast,
    RandFlip,
    RandGaussianNoise,
    RandAffine,
    RandRotate,
    RandShiftIntensity,
)

import config
import data_preproc.data_preproc_config as data_preproc_config
from data_preproc.data_preproc_ct_segmentation_map import get_segmentation_mask


def preprocess_inputs(inputs):
    """
    This function contains data preprocessing operations for each channel in 'inputs', which is CT, RTDOSE and
        segmentation_maps.

    IMPORTANT NOTE: no random transforms are allowed in this function, because this function will also be used for
        validation data!

    Source: https://docs.monai.io/en/latest/transforms.html?highlight=scaleintensity#scaleintensityrange

    Args:
        inputs (Numpy array): shape (batch_size, num_inputs, num_slices, num_rows, num_columns)

    Returns:

    """

    return inputs


def preprocess_features(features):
    """
    This function contains data preprocessing operations for 'features', which contains xerostomia baseline and
        mean dose contra-lateral parotid gland.

    IMPORTANT NOTE: no random transforms are allowed in this function, because this function will also be used for
        validation data!

    Source: https://docs.monai.io/en/latest/transforms.html?highlight=scaleintensity#scaleintensityrange

    Args:
        features (Numpy array): shape (batch_size, num_features)

    Returns:

    Args:
        features:

    Returns:

    """

    return features


def data_aug_transforms(modes, p, strength, seed):
    """
    Data augmentation transforms.

    Source: https://github.com/Project-MONAI/tutorials/blob/master/modules/3d_image_transforms.ipynb
    Source: https://docs.monai.io/en/latest/transforms.html?highlight=Rand3DElastic#monai.transforms.Rand3DElastic

    Note: MONAI documentation describes that the transformers require 3D data to have shape (nchannels, H, W, D), but
        shapes (1, H, W, D) and (H, W, D) results in same augmented data, i.e. shape (H, W, D) is fine.

    Args:
        modes (list): interpolation modes, either 'bilinear' (CT and RTDOSE) or 'nearest' (Segmentation)
        # range_value: original_max_value - original_min_value, required for fair RandGaussianNoise()
        p: probability
        strength: strength of data augmentation: strenght=0 is no data aug (except flipping)
        seed: set seed for transform for reproducibility

    Returns:

    """
    pass


def data_aug(arr, p, strength, seed):
    """
    Perform data augmentation on CT, RTDOSE and Segmentation.

    Args:
        arr: Tensor with shape (3, depth, height, width)
        p: probability
        strength: strength of data augmentation: strenght=0 is no data aug (except flipping)
        seed:

    Returns:

    """

    return arr


def flip(arr, mode, strength, seed):
    augmenter = RandFlip(prob=1.0, spatial_axis=-1)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)


def translate(arr, mode, strength, seed):
    # augmenter = RandAffine(prob=1.0, translate_range=(7 * strength, 7 * strength, 7 * strength),
    #                        padding_mode='border', mode=mode),  # 3D: (num_channels, H, W[, D])
    augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                              translate_range=(round(7 * strength), round(7 * strength), round(7 * strength)),
                              padding_mode='border', mode=mode)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)


def rotate(arr, mode, strength, seed):
    # augmenter = RandRotate(prob=1.0, range_x=(np.pi / 24) * strength, align_corners=True,
    #                        padding_mode='border', mode=mode),
    augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                              rotate_range=((np.pi / 24) * strength, (np.pi / 24) * strength, (np.pi / 24) * strength),
                              padding_mode='border', mode=mode)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)


def scale(arr, mode, strength, seed):
    # augmenter = RandAffine(prob=1.0, scale_range=(0.07 * strength, 0.07 * strength, 0.07 * strength),
    #                        padding_mode='border', mode=mode),  # 3D: (num_channels, H, W[, D])
    augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                              scale_range=(0.07 * strength, 0.07 * strength, 0.07 * strength),
                              padding_mode='border', mode=mode)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)


def gaussian_noise(arr, mode, strength, seed):
    augmenter = RandGaussianNoise(prob=1.0, mean=0.0, std=0.02)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)


def intensity(arr, mode, strength, seed):
    # augmenter = RandShiftIntensity(prob=1.0, offsets=(0, 0.05 * strength))
    augmenter = RandAdjustContrast(prob=1.0, gamma=(0.9, 1))
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)


# https://docs.monai.io/en/latest/transforms.html?highlight=RandShiftIntensity#randshiftintensity
# https://docs.monai.io/en/latest/transforms.html?highlight=RandShiftIntensity#spatial
aug_list = [flip, translate, rotate, scale]


# augmentations = augmentations_labels  #  + [gaussian_noise, intensity]


def aug_mix(arr, mixture_width, mixture_depth, augmix_strength, device, seed):
    """
    Perform AugMix augmentations and compute mixture.

    Source: https://github.com/google-research/augmix/blob/master/augment_and_mix.py

    Args:
        arr: (preprocessed) Numpy array of size (3, depth, height, width)
        mixture_width: width of augmentation chain (cf. number of parallel augmentations)
        mixture_depth: range of depths of augmentation chain (cf. number of consecutive augmentations)
            OLD: -1 enables stochastic depth uniformly from [1, 3]
        augmix_strength:
        device:
        seed:

    Returns:
        mixed: Augmented and mixed image.
    """

    # OLD
    """
    # random.seed(a=seed)
    # np.random.seed(seed=seed)

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))
    # m = np.random.uniform(0, 0.25)

    mix = torch.zeros_like(arr)
    for i in range(mixture_width):
        image_aug = arr.clone()

        # OLD
        # depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
        depth = np.random.randint(mixture_depth[0], mixture_depth[1] + 1)
        for _ in range(depth):
            # op = np.random.choice(aug_list)
            idx = random.randint(0, len(aug_list) - 1)
            op = aug_list[idx]
            seed_i = random.getrandbits(32)

            # CT
            image_aug[0] = op(arr=image_aug[0], mode=config.ct_interpol_mode_2d, strength=augmix_strength, seed=seed_i)
            # RTDOSE
            image_aug[1] = op(arr=image_aug[1], mode=config.rtdose_interpol_mode_2d, strength=augmix_strength, seed=seed_i)
            # Segmentation
            image_aug[2] = op(arr=image_aug[2], mode=config.segmentation_interpol_mode_2d, strength=augmix_strength,
                              seed=seed_i)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = (1 - m) * arr + m * mix
    """
    ws = torch.tensor(np.random.dirichlet([1] * mixture_width), dtype=torch.float32)
    m = torch.tensor(np.random.beta(1, 1), dtype=torch.float32)
    # m = np.random.uniform(0, 0.25)

    mix = torch.zeros_like(arr, device=device)
    for i in range(mixture_width):
        image_aug = arr.clone()

        # OLD
        # depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
        depth = np.random.randint(mixture_depth[0], mixture_depth[1] + 1)
        for _ in range(depth):
            # op = np.random.choice(aug_list)
            idx = random.randint(0, len(aug_list) - 1)
            op = aug_list[idx]
            seed_i = random.getrandbits(32)

            # CT
            image_aug[0] = op(arr=image_aug[0], mode=config.ct_interpol_mode_2d, strength=augmix_strength, seed=seed_i)
            # RTDOSE
            image_aug[1] = op(arr=image_aug[1], mode=config.rtdose_interpol_mode_2d, strength=augmix_strength,
                              seed=seed_i)
            # Segmentation
            image_aug[2] = op(arr=image_aug[2], mode=config.segmentation_interpol_mode_2d, strength=augmix_strength,
                              seed=seed_i)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = (1 - m) * arr + m * mix

    return mixed



