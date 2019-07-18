'''
J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller.
Striving for Simplicity: The All Convolutional Net,
https://arxiv.org/abs/1412.6806
accepted iclr 2015
'''

class GuidedBackpropagation():
    def __init__(self, model):
        self.model = model
        self.model.eval()

if __name__ == "__main__":
    # get needed imformation
    origin_img, pretrained_model = get_input()

    # GuidedBackprop modeling
    guided_model = GuidedBackprop()
    guided_back_tensor = guided_model.get_gradient()

    # save image
    save_image(guided_back_tensor, save_path="")








































import torch
from torch.nn import ReLU
from torchvision import models

class GuidedBackprop():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()
        self.update_relus()

    def hook_layers(self):
        return

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            return
        def relu_forward_hook_function(module, ten_in, ten_out):
            return
        for pos, module in pretrained_model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # forward pass
        model_output = self.model(input_image)
        # zero gradients
        self.model.zero_grad()
        # target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1])
        one_hot_output[0][target_class] = 1
        # backward pass
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def get_input():
    return

def save_image(tensor, save_path):
    return

