import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


img_path = "rgb_square.png"


def load_img(path):
    # load image from path and convert to float
    img = plt.imread(path).astype(np.float32)
    if img.shape[2] == 4:
        # remove alpha channel
        img = img[:, :, :3]
    return img


def convert_rgb_to_hsv(img):
    # convert image to hsv
    hsv_img = img
    hsv_img = matplotlib.colors.rgb_to_hsv(hsv_img)
    return hsv_img


def convert_hsv_to_rgb(img):
    # convert image to rgb
    rgb_img = img
    rgb_img = matplotlib.colors.hsv_to_rgb(rgb_img)
    return rgb_img


def plot(orig_img_path, predicted_img):
    orig_img = load_img(orig_img_path)

    plt.subplot(3, 1, 1).set_title("RGB")
    plt.imshow(orig_img)

    plt.subplot(3, 1, 2).set_title("RGB>(MPL)>HSV>(MPL)>RGB")
    control_img = convert_hsv_to_rgb(convert_rgb_to_hsv(orig_img))
    plt.imshow(control_img)

    plt.subplot(3, 1, 3).set_title("RGB>(NN)>HSV>(MPL)>RGB")
    result_img = convert_hsv_to_rgb(predicted_img)
    plt.imshow(result_img)

    plt.tight_layout()
    plt.show()



