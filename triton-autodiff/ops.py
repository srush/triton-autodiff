import tangent
from tangent.grads import adjoint
import triton.language as tl
import triton
d = {}
nd = [tl.arange, tl.zeros, tl.full, tl.static_range,
      tl.randint, tl.rand, tl.randn, tl.randint4x]
@adjoint(tl.sum)
def sum(z, x):
    d[x] = tl.ones_like(x) * d[z]

@adjoint(tl.log)
def log(z, x):
    d[x] = (d[x] * d[z]) / x

@adjoint(tl.exp)
def exp(z, x):
    d[x] = d[x] * tl.exp(x) * d[z]

@adjoint(tl.abs)
def abs(z, x):
    d[x] = d[x] * tl.exp(x) * d[z]
@adjoint(tl.exp)
def exp(z, x):
    d[x] = d[x] * tl.exp(x) * d[z]

#@triton.jit(interpret=True)
def test(X, Y):
  x = tl.load(X + tl.arange(0, 10))
  y = tl.log(x)
  tl.store(Y + tl.arange(0, 10), y)
import torch
tangent.grad(test, verbose=1)
#x, y = torch.ones(10).cpu(), torch.zeros(10).cpu()
#test[(1,)](x, y)
# def f(x):
#   return tl.sum(x)