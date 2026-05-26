import torch

IMAGE_HEIGHT = 218
IMAGE_WIDTH = 178


def normalize_landmarks(target):
    landmarks = target.to(torch.float32).view(5, 2).clone()
    landmarks[:, 0] /= IMAGE_WIDTH
    landmarks[:, 1] /= IMAGE_HEIGHT
    return landmarks.reshape(-1)


def denormalize_landmarks(target):
    landmarks = target.to(torch.float32).view(5, 2).clone()
    landmarks[:, 0] *= IMAGE_WIDTH
    landmarks[:, 1] *= IMAGE_HEIGHT
    return landmarks.reshape(-1)
