'''
J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller.
Striving for Simplicity: The All Convolutional Net,
https://arxiv.org/abs/1412.6806
accepted iclr 2015
'''

import os
import argparse
from utils import get_input, save_image

import torch
from torch.nn import ReLU

class GuidedBackprop():
    def __init__(self, model):
        self.model = model
        self.gradient = None
        self.forward_relu_output = []
        self.model.eval()
        self.update_relus()
        self.hook_layer()

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            self.gradient = grad_in[0]
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_forward_hook_function(module, input, output):
            self.forward_relu_output.append(output)

        def relu_backward_hook_function(module, grad_input, grad_output):
            corresponding_forward_output = self.forward_relu_output[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_input[0], min=0.0)
            del self.forward_relu_output[-1]
            return (modified_grad_out,)

        for position, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_forward_hook(relu_forward_hook_function)
                module.register_backward_hook(relu_backward_hook_function)

    def get_gradient(self, input_image, target_class):
        # forward pass
        model_output = self.model(input_image)
        target = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        target[0][target_class] = 1

        # backward pass
        self.model.zero_grad()
        model_output.backward(gradient=target)
        gradients_as_arr = self.gradient.data.numpy()[0]
        return gradients_as_arr

if __name__ == "__main__":
    # get input index
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cat")
    args = parser.parse_args()
    target_image_idx = 0 if args.category == "cat" else 1
    # target_image_idx = 0
    # get needed imformation
    img, target_class, pretrained_model = get_input(target_image_idx)

    # GuidedBackprop modeling
    guided_model = GuidedBackprop(pretrained_model)
    guided_backprop_tensor = guided_model.get_gradient(img, target_class)

    # save image
    save_image(guided_backprop_tensor, save_path="saved_image", save_name="GB_"+args.category)
