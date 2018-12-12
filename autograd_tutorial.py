# -*- coding: utf-8 -*-
"""
Autograd: Automatic Differentiation
===================================

Central to all neural networks in PyTorch is the ``autograd`` package.
Let’s first briefly visit this, and we will then go to training our
first neural network.


The ``autograd`` package provides automatic differentiation for all operations
on Tensors. It is a define-by-run framework, which means that your backprop is
defined by how your code is run, and that every single iteration can be
different.

Let us see this in more simple terms with some examples.

Tensor
--------

``torch.Tensor`` is the central class of the package. If you set its attribute
``.requires_grad`` as ``True``, it starts to track all operations on it. When
you finish your computation you can call ``.backward()`` and have all the
gradients computed automatically. The gradient for this tensor will be
accumulated into ``.grad`` attribute.

To stop a tensor from tracking history, you can call ``.detach()`` to detach
it from the computation history, and to prevent future computation from being
tracked.

To prevent tracking history (and using memory), you can also wrap the code block
in ``with torch.no_grad():``. This can be particularly helpful when evaluating a
model because the model may have trainable parameters with `requires_grad=True`,
but for which we don't need the gradients.

There’s one more class which is very important for autograd
implementation - a ``Function``.

``Tensor`` and ``Function`` are interconnected and build up an acyclic
graph, that encodes a complete history of computation. Each tensor has
a ``.grad_fn`` attribute that references a ``Function`` that has created
the ``Tensor`` (except for Tensors created by the user - their
``grad_fn is None``).

If you want to compute the derivatives, you can call ``.backward()`` on
a ``Tensor``. If ``Tensor`` is a scalar (i.e. it holds a one element
data), you don’t need to specify any arguments to ``backward()``,
however if it has more elements, you need to specify a ``gradient``
argument that is a tensor of matching shape.
"""

import torch

###############################################################
# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# Do an operation of tensor:
y = x + 2
print(y)

###############################################################
# ``y`` was created as a result of an operation, so it has a ``grad_fn``.
print(y.grad_fn)

###############################################################
# Do more operations on y
z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``False`` if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###############################################################
# Gradients
# ---------
# Let's backprop now
# Because ``out`` contains a single scalar, ``out.backward()`` is
# equivalent to ``out.backward(torch.tensor(1))``.

out.backward()

###############################################################
# print gradients d(out)/dx
#

print(x.grad)

###############################################################
# You should have got a matrix of ``4.5``. Let’s call the ``out``
# *Tensor* “:math:`o`”.
# We have that :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` and :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# Therefore,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, hence
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# You can do many crazy things with autograd!

torch.manual_seed(0)
x = torch.randn(3, requires_grad=True)
print('x: {}'.format(x))

#y = (x*2).clone().detach().requires_grad_(True)
y = x*2
cnt = 0
while y.data.norm() < 1000:
    #print('cnt {}, norm of y: {}'.format(cnt, y.data.norm()))
    cnt += 1
    y = y * 2
   
print('y: {}'.format(y))
print('final norm of y: {}'.format(y.data.norm()))

###############################################################
#
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients, retain_graph=True) # no need to care of the layers beyound y if gradients are given already

print('y: {}'.format(y))
#print('grad on y: {}'.format(y.grad.data))
print('x: {}'.format(x))
print('grad on x: {}'.format(x.grad.data))

###############################################################
# You can also stop autograd from tracking history on Tensors
# with ``.requires_grad=True`` by wrapping the code block in
# ``with torch.no_grad()``:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# **Read Later:**
#
# Documentation of ``autograd`` and ``Function`` is at
# https://pytorch.org/docs/autograd
