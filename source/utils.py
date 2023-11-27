import random

import torch
import numpy as np
from tqdm import trange
from imageio import imwrite
import os


def generate_imgs(net_G, device, z_dim=128, size=5000, batch_size=128):
    net_G.eval()
    imgs = []
    with torch.no_grad():
        for start in trange(0, size, batch_size,
                            desc='Evaluating', ncols=0, leave=False):
            end = min(start + batch_size, size)
            z = torch.randn(end - start, z_dim).to(device)
            imgs.append(net_G(z).cpu())
    net_G.train()
    imgs = torch.cat(imgs, dim=0)
    imgs = (imgs + 1) / 2
    return imgs


def infiniteloop(dataloader):
    while True:
        for x, _ in iter(dataloader):
            yield x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_gradients(self):
    grads = []
    for param in self.parameters():
        if param.grad is not None:
            grad = param.grad
        else:
            grad = torch.zeros(param.shape, dtype=param.dtype, device=param.device)
        grads.append(grad.view(-1))
    grads = torch.cat(grads).clone()
    return grads

def reduce_grad(grad):
    grad = grad.norm(2, dim=-1)
    if grad.dim() > 0:
        grad = grad.sum(dim=0)
    return grad


def save_img(img_tensor, dir_path, img_name):
    """Saves a data point as .png file in dir_path with img_name as name.

    Args:
        img_tensor (torch.Tensor): The image of shape CxHxW in the range [0-1]
        dir_path (str): The folder where in which the images must be saved
        ig_name (str): The name to apply to the file containing the image.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"--> Created folder {dir_path}. Images will be saved here")

    img = 255.0 * torch.movedim(img_tensor, 0, 2).cpu().detach().numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, repeats=3, axis=-1)

    img = img.astype("uint8")
    imwrite(os.path.join(dir_path, f"{img_name}"), img)
