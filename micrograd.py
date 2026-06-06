import math
import numpy as np


class Value:
    
    def __init__(self,data,_children=(),_op=(),label=''):
        self.data = data
        self.grad = 0.0 # assuming that every weight initially doesnt impact the loss or final value
        self._backward =lambda: None # function that will be called to propagate the gradient backward
        self._prev = set(_children)
        self._op = _op
        self.label = label
        

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+') # new output Value object has data equal to the added output and children equal to the Value Objects themselves
        
        def _backward():
            self.grad += 1.0 * out.grad # how much does the output change with respect to self? 1.0 because if we change self by 1 unit, the output changes by 1 unit
            other.grad += 1.0 * out.grad # how much does the output change with respect to other? 1.0 because if we change other by 1 unit, the output
        out._backward = _backward
        return out
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rmul__(self, other): # other * self
        return self * other
    
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')

        def _backward():
            self.grad += other.data * out.grad # how much does the output change with respect to self? other.data because if we change self by 1 unit, the output changes by other.data units
            other.grad += self.data * out.grad # how much does the output change with respect to other? self.data because if we change other by 1 unit, the output changes by self.data units
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward
        return out
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad # how much does the output change with respect to self? 1 - t^2 because if we change self by 1 unit, the output changes by 1 - t^2 units
        out._backward = _backward
        return out
    def backward(self):
    
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


