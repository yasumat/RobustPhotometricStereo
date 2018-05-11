import cv2
import glob
import numpy as np


def load_lighttxt(filename):
    """
    Load light file specified by filename.
    The format of lights.txt should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.txt
    :return: light matrix (3 \times f)
    """
    Lt = np.loadtxt(filename)
    return Lt.T


def load_lightnpy(filename):
    """
    Load light numpy array file specified by filename.
    The format of lights.npy should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.npy
    :return: light matrix (3 \times f)
    """
    Lt = np.load(filename)
    return Lt.T

def load_image(filename):
    """
    Load image specified by filename (read as a gray-scale)
    :param filename: filename of the image to be loaded
    :return img: loaded image
    """
    return cv2.imread(filename, 0)


def load_images(foldername, ext):
    """
    Load images in the folder specified by the "foldername" that have extension "ext"
    :param foldername: foldername
    :param ext: file extension
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*." + ext)):
        im = cv2.imread(fname, 0)
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def disp_normalmap(N, height, width, delay=0):
    """
    Visualize normal map
    :param N: normal map (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :return: None
    """
    if N is None:
        raise ValueError("Surface normal N is None")
    N = np.reshape(N, (height, width, 3))  # Reshape to image coordinates
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    cv2.imshow('normal map', N)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()
