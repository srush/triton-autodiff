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

def check(tr1, tr2, c, x_shape=(16,), y_shape=(16,)):
    x = torch.ones(*x_shape, requires_grad=True, dtype=float, device=torch.device(0))
    y = torch.zeros(*y_shape, dtype=float).cuda()
    tr1[(1,)](x, y)
    y2 = c(x)
    assert torch.allclose(y, y2)
    y2.sum().backward()
    x_grad = torch.zeros(*x_shape, dtype=float).cuda()
    y_grad = torch.ones(*y_shape, dtype=float).float().cuda()
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
    check(*make_pair("exptt", "expbacktt"), exp_check)

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
    check(*make_pair("logtt", "logbacktt"), log_check)


# Complex
def comp(x):
    return tl.log(x) * tl.exp(x)
comptt = triton.jit(comp)

def compback(x, dx):
    pass
compbacktt = grad(comp, compback)

def comp_check(x):
    return torch.log(x) * torch.exp(x)

def test_run3():
    check(*make_pair("comptt", "compbacktt"), comp_check)

@triton.jit
def ub1(X, Y):
  r = tl.arange(0, 16)
  r2 = tl.arange(0, 32)
  x = tl.load(X + 16 * r2[:, None] + r)
  y = triton_unbroadcast(x, tl.arange(0, 16).shape)
  tl.store(Y + r, y)

@triton.jit
def ub2(X, Y):
  r = tl.arange(0, 16)
  r2 = tl.arange(0, 32)
  x = tl.load(X + 16 * r2[:, None] + r)
  y = triton_unbroadcast(x, tl.arange(0, 32)[:, None].shape)
  tl.store(Y + r2[:, None], y)


def test_unbroadcast():
    x = torch.ones(32, 16, requires_grad=True, device=torch.device(0))
    y = torch.zeros(16, requires_grad=True, device=torch.device(0))
    ub1[(1,)](x, y)
    assert torch.allclose(x.sum(0), y)
    y = torch.zeros(32, requires_grad=True, device=torch.device(0))
    ub2[(1,)](x, y)
    assert torch.allclose(x.sum(1), y)
# broadcast


def comp2(x):
    return tl.expand_dims(x, 1) * x
comp2tt = triton.jit(comp2)

def comp2back(x, dx):
    pass
comp2backtt = grad(comp2, comp2back)

def comp2_check(x):
    return x[:, None] * x

@triton.jit
def dcomp2dx(x, b_return):
    _return2 = tl.expand_dims(x, 1)
    bx = zeroslike(x)
    b_return2 = zeroslike(_return2)

    # Grad of: _return = _return2 * x
    _b_return2 = triton_unbroadcast(b_return * x, _return2.shape)
    # _bx2 = triton_unbroadcast(b_return * _return2, x)
    # b_return2 = add_grad(b_return2, _b_return2)
    # bx = add_grad(bx, _bx2)

    # # Grad of: _return2 = tl.expand_dims(x, 1)
    # _bx = tl.sum(b_return2, 1)
    # bx = add_grad(bx, _bx)
    return bx


@triton.jit
def tr1(X, Y):
  r = tl.arange(0, 16)
  x = tl.load(X + r)
  y = comp2tt(x)
  tl.store(Y + 16 * r[:, None] + r, y)

@triton.jit
def tr2(X, dX, dY):
  r = tl.arange(0, 16)

  r2 = tl.arange(0, 16)[:, None]
  x = tl.load(X + r)
  dy = tl.load(dY + 16 * r2 + r)
  tl.static_print("shape", dy.shape)
  dx = dcomp2dx(x, dy)
  tl.static_print("shape", dx.shape)
  tl.store(dX + r, dx)

def test_run4():
    check(tr1, tr2, comp2_check, x_shape=(16,), y_shape=(16, 16))
