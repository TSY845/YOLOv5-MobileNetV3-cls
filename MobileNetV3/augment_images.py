import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2


def random_rotation(image, rotation_range):
    """随机旋转图像"""
    angle = random.uniform(-rotation_range, rotation_range)
    return image.rotate(angle)


def random_shift(image, shift_range):
    """随机平移图像"""
    x_shift, y_shift = np.random.uniform(-shift_range, shift_range, size=2)
    return image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))


def random_zoom(image, zoom_range):
    """随机缩放图像"""
    zoom_factor = random.uniform(*zoom_range)
    w, h = image.size
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    if zoom_factor < 1.0:
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        x2 = x1 + w
        y2 = y1 + h
        image = image.crop((x1, y1, x2, y2))
    else:
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        x2 = x1 + new_w
        y2 = y1 + new_h
        black_image = Image.new(image.mode, (w, h), color='black')
        black_image.paste(image, (x1, y1, x2, y2))
        image = black_image
    return image


def random_noise(image, noise_range):
    """随机加噪声"""
    noise_factor = random.uniform(*noise_range)
    if noise_factor > 0.0:
        image_array = np.array(image)
        noise = np.random.normal(scale=noise_factor, size=image_array.shape).astype(np.int32)
        image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array)
    return image


def augment_images(input_folder, output_folder, augment_num=10):
    """增强图片并保存"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义增强参数
    rotation_range = 5  # 旋转角度范围
    shift_range = 0.1  # 平移范围，占图片宽度和高度的比例
    zoom_range = (0.9, 1.1)  # 缩放范围
    noise_range = (0, 20)  # 噪声范围，最大值是 255
    print("Start augmenting...")

    # 遍历文件夹中的图片文件并增强
    for filename in os.listdir(input_folder):
        # 读取图片
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # 对每张图片进行 augment_num 次增强并保存
        for i in range(augment_num):
            # 随机旋转、平移、缩放、加噪声
            # img_aug = random_rotation(img, rotation_range)
            img_aug = random_shift(img, shift_range)
            img_aug = random_zoom(img_aug, zoom_range)
            # img_aug = random_noise(img_aug, noise_range)

            # 保存增强后的图片
            output_filename = os.path.splitext(filename)[0] + f"_aug_{i}_" + os.path.splitext(filename)[1]
            output_path = os.path.join(output_folder, output_filename)
            img_aug.save(output_path)


if __name__ == "__main__":
    input_folder = "E:\\Intern\\Dataset\\fire_door\\VID_20230412_104238"
    output_folder = "E:\\Intern\\Dataset\\fire_door\\closed_augmented"
    augment_images(input_folder, output_folder, 2)