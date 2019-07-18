def get_input():
    return

def save_image(tensor, save_path):
    return

from torchvision import models
from PIL import Image
pretrained_model = models.alexnet(pretrained=True)

example_list = (('input_image/cat.jpg', 285),
                ('input_image/castle.jpg', 483))
example_index = 1
img_path = example_list[example_index][0]
target_class = example_list[example_index][1]
file_name_to_export = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
# Read image
original_image = Image.open(img_path).convert('RGB')

torch.argmax(pretrained_model(preprocess_image(original_image)))

import numpy as np
import torch
from torch.autograd import Variable
def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var