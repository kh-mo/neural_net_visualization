'''
pytorch는 network를 통과하는 forward path와 backward path를 hooking하여 변형, 사용이 가능하다
pretrained_model의 중간 모듈을 낚아채어(hooking) 사용하는 방법은 아래에 코드를 보며 이해할 수 있다
               -----
              |  m  |
    -- a -->  |  o  |  -- b -->
              |  d  |
              |  u  |
    <-- d --  |  l  |  <-- c --
              |  e  |
               -----

J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller.
Striving for Simplicity: The All Convolutional Net,
https://arxiv.org/abs/1412.6806

pytorch 기본 설정은 위 페이퍼 figure 1의 backpropagation 부분에 해당한다
'''

import torch
from utils import get_input
from torch.nn import ReLU, Linear
import numpy as np

target_image_idx = 0
img, target_class, pretrained_model = get_input(target_image_idx)

forward_input_list = []
forward_output_list = []
backward_input_list = []
backward_output_list = []

# forward path 진행 시, 입력과 출력을 hooking할 수 있다
def forward_hook_function(module, grad_input, grad_output):
    forward_input_list.append(grad_input) # a에 해당한다
    forward_output_list.append(grad_output) # b에 해당한다

# backward path 진행 시, 입력과 출력을 hooking할 수 있다
# backward는 input과 output에서 혼동이 올 수 있으니 주의가 필요하다
def backward_hook_function(module, grad_input, grad_output):
    backward_input_list.append(grad_input) # d에 해당한다
    backward_output_list.append(grad_output) # c에 해당한다

for position, module in pretrained_model.classifier._modules.items():
    if isinstance(module, Linear) or isinstance(module, ReLU):
        '''
        Tip!!
        Relu module의 경우 d의 결과값은 c의 결과값에 b를 element_wise multiplication을 한 것이다
        이 때 b는 양수일 경우 1로 변환된다(relu의 양수부분 기울기를 의미)
        '''
        module.register_forward_hook(forward_hook_function)
        module.register_backward_hook(backward_hook_function)

model_output = pretrained_model(img)
target = torch.FloatTensor(1, model_output.size()[-1]).zero_()
target[0][target_class] = 1

# backward pass
pretrained_model.zero_grad()
model_output.backward(gradient=target)
