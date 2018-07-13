# -*- coding: utf-8 -*-
"""
    Given the data augmentation arg, modify the pixels in the given image.
    Images are read using opencv in the original implementation,
    the color channels are in BGR order. Moreover, if the pyroch pre-trained
    models are used, then the color channels are in RGB order.
"""

import cv2
import numpy as np
import PIL
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms.functional as F
import random

__all__ = ['']
__author__ = 'Yuan Chen'
__copyright__ = '2018 LAMDA'
__date__ = '2018-07-04'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-07-13'
__version__ = '1.0'


def image_gamma_correction(img, gamma = 1.0, use_random = False):
    """Given an image, use gamma_correction to adjust the brightness of images.

    Param:
        - gamma: the gamma value for adjustment
    Return:
        - image in BGR order
    """
    # assert gamma > 0.0
    # invGamma = 1.0 / gamma
    # table = np.array([((((i + 0.5) / 256) ** invGamma) * 256 - 0.5) for i in range(0, 256)]).astype("uint8")
    # return cv2.LUT(img, table)

    if gamma != 1.0:
        # convert opencv image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # apply torchvision.transform
        if use_random:
            gamma = random.uniform(max(0.1, 1 - gamma), 1 + gamma)
            pil_img = F.adjust_gamma(pil_img, gamma)
        else:
            pil_img = F.adjust_gamma(pil_img, gamma)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img

def image_change_brightness(img, brightness = 0, use_random = False):
    """Given an image, transform the brightness of the images.
    Param:
        - brightness: In F.adjust_brightness, the value of brightness will give
        the followig result:
            0 will give a black image
            1 will give the original image
            2 will increase the brightness of the image by the factor of 2
        - random: determine whether we need to randomly generate a brightness
        factor in the valid range.
    Return:
        - image in BGR order
    """
    if brightness > 0:
        # convert opencv image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # randomly assign
        if use_random:
            brightness = random.uniform(max(0.3, 1 - brightness), 1 +
                    brightness)
            pil_img = F.adjust_brightness(pil_img, brightness)
        else:
            pil_img = F.adjust_brightness(pil_img, brightness)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img

def image_change_saturation(img, saturation = 0, use_random = False):
    """Instead of converting RGB to HSV or HSL and adjusting the S channel of
    the images, using the algorithm from Photoshop to implement this part.

    Param:
        - saturation: In F.adjust_saturation, the value of saturation will give
        the followig result:
            0 will give a black and white image
            1 will give the original image
            2 will increase the saturation of the image by the factor of 2
        - random: determine whether we need to randomly generate a saturation
        factor in the valid range.
    Return:
        - image in BGR order
    """
    if saturation > 0:
        # convert opencv image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if use_random:
            saturation = random.uniform(max(0, 1 - saturation), 1 + saturation)
            pil_img = F.adjust_saturation(pil_img, saturation)
        else:
            pil_img = F.adjust_saturation(pil_img, saturation)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img

def image_change_contrast(img, contrast = 0, use_random = False):
    """Given the image, change the contrast of the image.

    Param:
        - contrast: In F.adjust_contrast, the value of contrast will give
        the followig result:
            0 will give a solid gray image
            1 will give the original image
            2 will increase the contrast of the image by the factor of 2
        - random: determine whether we need to randomly generate a contrast
        factor in the valid range.
    Return:
        - image in BGR order
    """
    if contrast > 0:
        # convert opencv image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if use_random:
            contrast = random.uniform(max(0, 1 - contrast), 1 + contrast)
            pil_img = F.adjust_contrast(pil_img, contrast)
        else:
            pil_img = F.adjust_contrast(pil_img, contrast)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img

def image_change_sharpness(img, sharpness = 0, use_random = False):
    """Given the image, change the sharpness of the image.

    Param:
        - sharpness: the value of contrast will give the following result:
            0 will give a blurred image
            1 will give the original image
            2 will enhance the sharpness of the image by the factor of 2
    Return:
        - image in BGR order
    """
    if sharpness > 0:
        # convert opencv image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if use_random:
            sharpness = random.uniform(max(0 ,1 - sharpness), 1 + sharpness)
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(sharpness)
        else:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(sharpness)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img

def image_gaussian_smoothing(img, kernel_size = 3, sigmaX = 0, use_random =
        False):
    """Given an image, use gaussian kernel to smooth the image to reduce the
    attention on the details.

    Param:
        - kernel_size: the size of the kernel
        - sigmaX: the standard derivation
        - use_random: random or not
    Return;
        - image in BGR order
    """

    if use_random:
        sigmaX = int(random.uniform(0, sigmaX))
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX)
    else:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX)

    return img

def image_gaussian_noise(img, mean = 0, variance = 0.1, use_random = False):
    """Given an image, add gaussian noise to the image

    Param:
        - mean: mean of the gaussian kernel
        - variance: variance of the gaussian kernel
        - use_random: random or not
    """
    if use_random:
        mean = random.uniform(0, mean)
        variance= random.uniform(0, variance)

    row, col, channel = img.shape
    sigma = variance ** 0.5
    gaussian_map = np.random.normal(mean, sigma, (row, col, channel))
    gaussian_map = gaussian_map.reshape(row, col, channel)
    noisy_img = np.array(img).astype(np.float32) + gaussian_map

    return np.array(noisy_img).astype(np.uint8)

def image_pca_jittering(img, mean = 0, variance = 0.1, use_random = False):
    """Given an image, perform pca jittering

    Param:
        - mean: mean of normal variate for alpha
        - variance: variance of normal variate for alpha
        - use_random: random or not
    Return:
        - image after pca jittering
    """

    if use_random:
        mean = random.uniform(0, mean)
        variance = random.uniform(0, variance)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = np.asarray(pil_img, dtype = "float32")
    pil_img = pil_img / 255.
    one_channel_size = pil_img.size // 3
    img1 = pil_img.reshape(one_channel_size, 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])
    lamda, p = np.linalg.eig(img_cov)

    p = np.transpose(p)

    alpha1 = random.normal(variate(mean, variance))
    alpha2 = random.normal(variate(mean, variance))
    alpha3 = random.normal(variate(mean, variance))

    v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))

    add_num = np.dot(p, v)

    img2 = np.array([pil_img[:, :, 0] + add_num[0], pil_img[:, :, 1] +
        add_num[1], pil_img[:, :, 2] + add_num[2]])
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)

    img = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    return img

def image_channel_shifting(img, intensity = 10, channel_idx = 0, use_random = False):
    """ Given an image, shift one channel by the intensity

    Param:
        - intensity: shift amount
        - channel_idx: channel idx, should be 0, 1, 2
        - use_random: random or not
    Return:
        - image after shifting a certain amount of intensity in one channel
    """

    assert channel_idx in [0, 1, 2]
    if use_random:
        channel_idx = random.randint(0, 2)

    img1 = np.rollaxis(img, channel_idx, 0)
    min_img1, max_img1 = np.min(img1), np.max(img1)

    channel_images = [np.clip(x_channel + np.random.uniform(-intensity,
        intensity), min_img1, max_img1) for x_channel in img1]

    img1 = np.stack(channel_images, axis = 0)
    img1 = np.rollaxis(img1, 0, channel_idx + 1)

    return np.array(img1).astype(np.uint8)

def image_saltnpepper_noise(img, sp_ratio = 0.5, amount_ratio = 0.1, use_random = False):
    """Given an image, add salt and pepper noise on the image

    Param:
        - sp_ratio: the ratio of salt over pepper over the image
        - amount_ratio: the ratio of noise that you want to add on the image
        - use_random: random or not
    Return:
        - the polluted image
    """
    assert sp_ratio >=0 and sp_ratio <= 1
    assert amount_ratio <= 0.5

    if use_random:
        sp_ratio = random.uniform(0, sp_ratio)
        amount_ratio = random.uniform(0, amount_ratio)
    img_out = np.copy(img)

    num_salt = np.ceil(amount_ratio * img.size * sp_ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    img_out[coords] = 1

    num_pepper = np.ceil(amount_ratio * img.size * (1 - sp_ratio))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    img_out[coords] = 0

    return img_out

def main():
    pass

if __name__ == "__main__":
    main()
