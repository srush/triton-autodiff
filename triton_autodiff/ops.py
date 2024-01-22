from __future__ import annotations

import tangent
import triton
import triton.language as tl
from tangent.grads import adjoint
import gast
from io import StringIO
import sys
import torch

tangent.utils.INIT_GRAD = tangent.quoting.quote("zeroslike")
tangent.utils.ADD_GRAD = tangent.quoting.quote("add_grad")


def shape_l(shape):
  return len(shape)

@triton.jit
def zeroslike(x):
  return tl.zeros(x.shape, tl.float32)

@triton.jit
def triton_unbroadcast(array, other):
   l: tl.constexpr = tl.constexpr(shape_l(array.shape))
   ol: tl.constexpr = tl.constexpr(shape_l(other.value))
   #tl.static_print("l", l, ol)
   for i in tl.static_range(0, l):
     #tl.static_print("i", i, array.shape, i >= ol, array.shape[l-(1 + i)] > other.value[ol-(1 + i)])
     if i >= ol:
         array = tl.sum(array, l-(1 + i))
         array = tl.expand_dims(array, l-(1 + i))
     elif array.shape[l-(1 + i)] > other.value[ol-(1 + i)]:

         array = tl.sum(array, l-(1 + i))
         array = tl.expand_dims(array, l-(1 + i))
     tl.static_assert(tl.constexpr(shape_l(array.shape)) == l)
   return tl.view(array, other.value)

# These are just copied from tangent

@adjoint(tl.log)
def log(y, x):
  d[x] = d[y] / x


@adjoint(tl.cos)
def cos(y, x):
  d[x] = -d[y] * tl.sin(x)


@adjoint(tl.sin)
def sin(y, x):
  d[x] = d[y] * tl.cos(x)


@adjoint(tl.exp)
def exp(y, x):
  d[x] = y * d[y]


@adjoint(tl.dot)
def dot(y, x1, x2):
  d[x1] = tl.trans(tl.dot(x2, tl.trans(d[y])))
  d[x2] = tl.dot(tl.trans(x1), d[y])

@adjoint(tl.sqrt)
def sqrt(y, x):
  d[x] = d[y] / (2.0 * y)


@adjoint(tl.expand_dims)
def expand_dims(y, x, axis):
  tl.static_assert(y.shape[axis] == 1)
  d[x] = tl.view(d[y], x.shape)

# Binary ops: z = op(x, y)
@adjoint(gast.Mult)
def mult(z, x, y):
  d[x] = triton_unbroadcast(d[z] * y, x.shape)
  d[y] = triton_unbroadcast(d[z] * x, y.shape)


@adjoint(gast.Add)
def add(z, x, y):
  d[x] = triton_unbroadcast(d[z], x.shape)
  d[y] = triton_unbroadcast(d[z], y.shape)


@adjoint(gast.Pow)
def pow(z, x, y):
  d[x] = y * x ** (y - 1) * d[z]
  d[y] = tl.log(x) * x ** y * d[z]


@adjoint(gast.Sub)
def sub(z, x, y):
  d[x] = triton_unbroadcast(d[z], x)
  d[y] = -triton_unbroadcast(d[z], y)


@adjoint(gast.Div)
def div(z, x, y):
  d[x] = d[z] / y
  d[y] = -d[z] * x / (y * y)


# Unary ops: y = op(x)
@adjoint(gast.USub)
def usub(y, x):
  d[x] = -d[y]


@adjoint(gast.UAdd)
def uadd(y, x):
  d[x] = d[y]

@triton.jit
def add_grad(left, right):
  right = triton_unbroadcast(right, left.shape)
  return left + right

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def grad(forward, back, wrt=(0,)):
  with Capturing() as output:
     tangent.grad(forward, wrt, verbose=1, check_dims=False)
  print("\n".join(output))
  exec("\n".join(output))

  newback = triton.JITFunction(back)
  newback.src = "\n".join(output)
  return newback

def check(tr1, tr2, c, x_shape=(16,), y_shape=(32,), z_shape=(32, 16)):
    x = torch.ones(*x_shape, requires_grad=True, device=torch.device(0))
    y = torch.ones(*y_shape, requires_grad=True, device=torch.device(0))
    z = torch.zeros(*z_shape).cuda()
    tr1[(1,)](x, y, z)
    z2 = c(x, y)
    assert torch.allclose(z, z2)
    z_grad = torch.rand(*z_shape).float().cuda()
    x_grad_1, y_grad_1 = torch.autograd.grad([z2], [x, y], grad_outputs=[z_grad])
    x_grad = torch.zeros(*x_shape).cuda()
    y_grad = torch.zeros(*y_shape).cuda()
    tr2[(1,)](x, y, x_grad, y_grad, z_grad)
    assert torch.allclose(x_grad_1, x_grad)
    assert torch.allclose(y_grad_1, y_grad)