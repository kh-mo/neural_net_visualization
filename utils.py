import numpy as np
from PIL import Image

import torch
from torchvision import models
from torch.autograd import Variable

def get_input(idx):
    example_list = (('input_image/cat.jpg', 285), ('input_image/castle.jpg', 483))
    img_path = example_list[idx][0]
    target_class = example_list[idx][1]

    # Read image
    img = preprocess_image(img_path)

    # Load pretrained net
    pretrained_model = models.alexnet(pretrained=True)
    return img, target_class, pretrained_model

def preprocess_image(img_path):
    original_image = Image.open(img_path).convert('RGB')
    original_image.thumbnail((512, 512))
    im_as_arr = np.float32(original_image)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def save_image(tensor, save_path):
    return

