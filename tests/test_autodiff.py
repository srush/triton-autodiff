import torch
import triton
import triton.language as tl
import tangent
from triton_autodiff import *
import pytest
def tre1(X, Y):
    pass
def tre2(X, dX, dY):
    pass

def make_pair( f1, f2):
    st1 = f"""
def tre1(X, Y):
  r = tl.arange(0, 16)
  x = tl.load(X + r)
  y = {f1}(x)
  tl.store(Y + r, y)
    """
    st2 = f"""
def tre2(X, dX, dY):
  r = tl.arange(0, 16)
  x = tl.load(X + r)
  dy = tl.load(dY + r)
  dx = {f2}(x, dy)
  tl.store(dX + r, dx)
    """
    t1 = triton.jit(tre1)
    t1.src = st1
    t2 = triton.jit(tre2)
    t2.src = st2
    return t1, t2

def check(tr1, tr2, c):
    x = torch.ones(16, requires_grad=True, device=torch.device(0))
    y = torch.zeros(16).cuda()
    tr1[(1,)](x, y)
    y2 = c(x)
    assert torch.allclose(y, y2)
    y2.sum().backward()
    x_grad = torch.zeros(16).cuda()
    y_grad = torch.ones(16).float().cuda()
    tr2[(1,)](x, x_grad, y_grad)
    assert torch.allclose(x.grad, x_grad)

# Exp
def exp(x):
    return tl.exp(x)
exptt = triton.jit(exp)

def expback(x, dx):
    pass
expbacktt = grad(exp, expback)

def exp_check(x):
    return torch.exp(x)

def test_run():
    check("exptt", "expbacktt", exp_check)

# Log
def log(x):
    return tl.log(x)
logtt = triton.jit(log)

def logback(x, dx):
    pass
logbacktt = grad(log, logback)

def log_check(x):
    return torch.log(x)

def test_run2():
    check("logtt", "logbacktt", log_check)


# Complex
def comp(x):
    return tl.log(x) * tl.exp(x)
comptt = triton.jit(comp)

def compback(x, dx):
    pass
compbacktt = grad(comp, compback)

@triton.jit
def tre1(X, Y):
  r = tl.arange(0, 16)
  x = tl.load(X + r)
  y = comptt(x)
  tl.store(Y + r, y)

@triton.jit
def tre2(X, dX, dY):
  r = tl.arange(0, 16)
  x = tl.load(X + r)
  dy = tl.load(dY + r)
  dx = compbacktt(x, dy)
  tl.store(dX + r, dx)

def comp_check(x):
    return torch.log(x) * torch.exp(x)

def test_run3():
    check(tre1, tre2, comp_check)
test_run3()