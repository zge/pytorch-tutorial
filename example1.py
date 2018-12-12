#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An example to show how the backprogation is computed in pytorch
reference: https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments

Zhenhao Ge, 2018-12-11
"""

from torch.autograd import Variable
import torch

x = Variable(torch.FloatTensor([1,2,3,4]), requires_grad=True)
#x = torch.tensor([1,2,3,4], dtype=torch.float, requires_grad=True) # alternatively
z = 2*x # indicate d(z)/d(x) = 2
out = z.sum(dim=0) # get sum along dim 0 (here is only 1d)

# set gradients on z (gradients on nodes other than 0th is 0s)
gradients = torch.FloatTensor([[1, 0, 0, 0]]) # d(out)/d(z)
# do backward propagation with gradients on z to get gradients on x
# (only apply to 0th element, since the other gradients are all 0s)
# d(out)/d(x) = (d(out)/d(z)) * (d(z)/d(x)) = (d(out)/d(z)) * 2
z.backward(gradients, retain_graph=True)
print('gradients on x: {}'.format(x.grad.data)) 
x.grad.data.zero_(); #remove gradient in x.grad, o.w it will be accumulated

# set gradients on z (gradients on nodes other than 1st is 0s)
gradients = torch.FloatTensor([[0, 2, 0, 0]])
# do backward propagation with gradients on z to get gradients on x
# (only apply to 1st element, since the other gradients are all 0s)
# d(out)/d(x) = (d(out)/d(z)) * (d(z)/d(x)) = (d(out)/d(z)) * 2
z.backward(gradients, retain_graph=True)
print('gradients on x: {}'.format(x.grad.data))
x.grad.data.zero_();

# do backward for all elements of z, with weight equal to the derivative of
# loss w.r.t z_1, z_2, z_3 and z_4
gradients = torch.FloatTensor([[1, 2, -1, -2]])
z.backward(gradients, retain_graph=True)
print('gradients on x: {}'.format(x.grad.data))
x.grad.data.zero_();

# or we can directly backprop using loss
# this will do backward propagation from out to the very beginning (through every layer)
out.backward(retain_graph=True) # equivalent to out.backward(torch.FloatTensor([1.0]))
print('gradients on x: {}'.format(x.grad.data))   