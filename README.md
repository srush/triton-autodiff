# Triton-Autodiff

<a href="https://colab.research.google.com/github/srush/triton-autodiff/blob/main/Triton_Autodiff.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


This library is a proof-of-concept of autodifferentiation for [Triton](https://github.com/openai/triton/) GPU Code using the [Tangent](https://github.com/google/tangent) source-to-source compiler. Remarkably this project is only roughly [50 LoC](https://github.com/srush/triton-autodiff/blob/main/triton_autodiff/ops.py). 

Here's how it works:

```python
# 1) Define a Triton mathematical function
def fn1(x, y):
    a = tl.exp(x) 
    b = tl.log(tl.expand_dims(y, 1))
    c = a + b
    return a * b + tl.dot(c, c)
fn1_tt = triton.jit(fn1)

# 2) Give Signature of its backwards (generated function will print out)
def fn1_back(x, y, dz):
    pass

# 3) Call Tangent
fn1back_tt = grad(fn1, fn1_back, wrt=(0, 1))
```

The library then outputs the internal code if you want to take a look and debug.

```python
def dfn1dxy(x, y, b_return=1.0):
    a = tl.exp(x)
    _b = tl.expand_dims(y, 1)
    b = tl.log(_b)
    c = a + b
    tl_dot_c_c = tl.dot(c, c)
    a_times_b = a * b
    bx = zeroslike(x)
    by = zeroslike(y)
    b_b = zeroslike(_b)
    bc = zeroslike(c)
    bb = zeroslike(b)
    ba = zeroslike(a)
    btl_dot_c_c = zeroslike(tl_dot_c_c)
    ba_times_b = zeroslike(a_times_b)

    # Grad of: c = a + b
    _ba_times_b = triton_unbroadcast(b_return, a_times_b.shape)
    _btl_dot_c_c = triton_unbroadcast(b_return, tl_dot_c_c.shape)
    ba_times_b = add_grad(ba_times_b, _ba_times_b)
    btl_dot_c_c = add_grad(btl_dot_c_c, _btl_dot_c_c)
    _ba2 = triton_unbroadcast(ba_times_b * b, a.shape)
    _bb2 = triton_unbroadcast(ba_times_b * a, b.shape)
    ba = add_grad(ba, _ba2)
    bb = add_grad(bb, _bb2)
    _bc = tl.trans(tl.dot(c, tl.trans(btl_dot_c_c)))
    _bc2 = tl.dot(tl.trans(c), btl_dot_c_c)
    bc = add_grad(bc, _bc)
    bc = add_grad(bc, _bc2)
    _ba = triton_unbroadcast(bc, a.shape)
    _bb = triton_unbroadcast(bc, b.shape)
    ba = add_grad(ba, _ba)
    bb = add_grad(bb, _bb)

    # Grad of: b = tl.log(tl.expand_dims(y, 1))
    _b_b = bb / _b
    b_b = add_grad(b_b, _b_b)
    __b = _b
    tl.static_assert(__b.shape[1] == 1)
    _by = tl.view(b_b, y.shape)
    by = add_grad(by, _by)

    # Grad of: a = tl.exp(x)
    _a = a
    _bx = _a * ba
    bx = add_grad(bx, _bx)
    return bx, by
```

You can also use the code directly in a full Triton program. 

```python
# Boilerplate load and forward
@triton.jit
def tr_forward(X, Y, Z):
  r = tl.arange(0, 16)
  r2 = tl.arange(0, 16)
  x = tl.load(X + r)
  y = tl.load(Y + r2)
  z = fn1_tt(x, y)
  tl.store(Z + 16 * r2[:, None] + r, z)

# Boilerplate load and backward 
@triton.jit
def tr_backward(X, Y, dX, dY, dZ):
  r = tl.arange(0, 16)
  r2 = tl.arange(0, 16)
  x = tl.load(X + r)
  y = tl.load(Y + r2)
  dz = tl.load(dZ + 16 * r2[:, None] + r)
  dx, dy = fn1back_tt(x, y, dz)
  tl.store(dX + r, dx)
  tl.store(dY + r2, dy)
```

Should give the same answer as PyTorch

```python
# Torch version for sanity check.
def torch_check(x, y):
    a = x.exp()
    b = y[:, None].log()
    c = a + b
    return a * b + c @ c

def test_run():
    check(tr_forward, tr_backward, torch_check, x_shape=(16,), y_shape=(16,), z_shape=(16, 16))
    print("check succeeded!")
test_run()
```

## Todos

- [ ] Support control flow -> for / if
- [ ] Automatically add more static assertions
- [ ] Support for reductions / unreduce
- [ ] Support for associative scan



