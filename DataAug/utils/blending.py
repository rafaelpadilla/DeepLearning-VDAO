import math
import os

import cv2
import numpy as np

import _init_paths
from generic_utils import euclidean_distance


def blur_measurement(image):
    if isinstance(image, str):
        assert os.path.isfile(image), f'It was not possible to load image {image}'
        image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channelR = image[:, :, 2]
    channelG = image[:, :, 1]
    channelB = image[:, :, 0]
    try:
        grayVar = cv2.Laplacian(gray, cv2.CV_64F)
        grayVar = grayVar.var()
        RVar = cv2.Laplacian(channelR, cv2.CV_64F).var()
        GVar = cv2.Laplacian(channelG, cv2.CV_64F).var()
        BVar = cv2.Laplacian(channelB, cv2.CV_64F).var()
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    return [RVar, GVar, BVar, grayVar]


def enlarge_mask(mask, iterations):
    inv_mask = 255 - mask
    se = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    enlarged_mask = cv2.erode(src=inv_mask, kernel=se, iterations=iterations)
    enlarged_mask_bin = enlarged_mask / 255
    diffMask = np.add(enlarged_mask, mask)
    diffMask_bin = diffMask / 255
    return enlarged_mask, enlarged_mask_bin.astype(np.uint8), diffMask, diffMask_bin.astype(
        np.uint8)


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))
    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def apply_transformations(image, scale_factor_x, scale_factor_y, rotation_angle, flip_horizontally):
    # Flip horizontally
    if flip_horizontally is True:
        image = cv2.flip(image, 0)
    # Rotate image counter-clockwise considering the angle (in degrees)
    image = rotate_image(image, rotation_angle)
    # Rescale image
    image = cv2.resize(image,
                       None,
                       fx=scale_factor_x,
                       fy=scale_factor_y,
                       interpolation=cv2.INTER_CUBIC)
    return image


def blend_image_into_mask(image, mask):
    # A paths were passed instead of loaded images
    if isinstance(image, str):
        assert os.path.isfile(image), f'Image could not be found in the path: {image}'
        image = cv2.imread(image)
    if isinstance(mask, str):
        assert os.path.isfile(mask), f'Mask image could not be found in the path: {mask}'
        mask = cv2.imread(mask)
    if mask.ndim == 1:
        mask = cv2.merge((mask, mask, mask))
    # Multiply mask by image so we have the object only
    mergedImage = np.multiply(image, mask / 255)
    mergedImage = mergedImage.astype(np.uint8)
    return mergedImage


def blend_iterative_blur(image,
                         mask,
                         background,
                         xIni=0,
                         yIni=0,
                         scale_factor_x=1,
                         scale_factor_y=1,
                         rotation_angle=0,
                         flip_horizontally=False):
    # Check if files exist
    if isinstance(image, str):
        assert os.path.isfile(image), f'Image could not be found in the path: {image}'
        image = cv2.imread(image)
    if isinstance(mask, str):
        assert os.path.isfile(mask), f'Mask image could not be found in the path: {mask}'
        mask = cv2.imread(mask)
    if isinstance(background, str):
        assert os.path.isfile(
            background), f'Background image could not be found in the path: {background}'
        background = cv2.imread(background)
    image = apply_transformations(image, scale_factor_x, scale_factor_y, rotation_angle,
                                  flip_horizontally)
    mask = apply_transformations(mask, scale_factor_x, scale_factor_y, rotation_angle,
                                 flip_horizontally)
    # Before rotating the mask, the values are either 0 or 255. After rotation, some of these values
    # are changed. Therefore, we need to threshold
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_inv_bin = ((255 - mask) / 255).astype(np.uint8)
    # Blend shoe with mask
    img_mask_blended = blend_image_into_mask(image, mask)
    # Obtain 2 layers:
    # layer_1 => layer between mask_bin and enlarged by 5 iterations
    # layer_2 => layer between enlarged by 7 iterations and 3 iterations
    _, mask_larger_2, _, _ = enlarge_mask(mask, 3)
    _, mask_larger_1_bin, _, layer_1 = enlarge_mask(mask, 5)
    layer_1 = 1 - layer_1
    _, mask_larger_3, _, _ = enlarge_mask(mask, 7)
    layer_2_bin = np.subtract(mask_larger_2, mask_larger_3)
    layer_2_bin_inv = 1 - layer_2_bin
    # It's necessary to get width and height once the image was rescaled
    rsz_rows, rsz_cols, rsz_channels = image.shape
    # Get blur level of the region of the background:
    min_x = xIni
    min_y = yIni
    max_x = xIni + rsz_cols
    max_y = yIni + rsz_rows
    # define background roi
    roi_background = background[min_y:max_y, min_x:max_x, :]
    blur_level_reference = blur_measurement(roi_background)
    # add image in the background
    background_with_image = np.multiply(roi_background, mask_inv_bin)
    background_with_image = background_with_image + img_mask_blended
    # get level of blur
    blur_level_background_with_img = blur_measurement(background_with_image)
    # measure distance between blur levels
    dist = euclidean_distance(blur_level_background_with_img, blur_level_reference)
    rsz_rows, rsz_cols, rsz_channels = img_mask_blended.shape
    layer_1_with_background = np.multiply(layer_1, roi_background)
    layer_2_with_background = np.multiply(layer_2_bin, roi_background)
    object_with_border_background = np.add(img_mask_blended, layer_1_with_background)
    background_with_large_object_mask = np.multiply(roi_background, mask_larger_1_bin)
    # Looping and bluring
    iteration = 0
    return_image = roi_background
    while True:
        blurred_R = cv2.GaussianBlur(object_with_border_background[:, :, 2], (3, 3), 0)
        blurred_G = cv2.GaussianBlur(object_with_border_background[:, :, 1], (3, 3), 0)
        blurred_B = cv2.GaussianBlur(object_with_border_background[:, :, 0], (3, 3), 0)
        blurred_object_background = cv2.merge((blurred_B, blurred_G, blurred_R))
        ghosty_background_with_object = np.add(background_with_large_object_mask,
                                               blurred_object_background)
        pre_background_with_object = np.multiply(ghosty_background_with_object, layer_2_bin_inv)
        final_image = layer_2_with_background + pre_background_with_object
        blur_level = blur_measurement(final_image)
        distLoop = euclidean_distance(blur_level, blur_level_reference)
        # Distance was increased or did not change, break loop
        if distLoop >= dist:
            break
        else:
            # Continue
            dist = distLoop
            iteration = iteration + 1
        return_image = final_image
    return return_image, [min_x, min_y, max_x, max_y]
