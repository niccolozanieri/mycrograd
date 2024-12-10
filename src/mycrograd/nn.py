from src.mycrograd.engine import Value
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = random.uniform(-1, 1)
    
    def __call__(self, x):
        out = 0
        for wi, xi in zip(self.w, x):
            out = wi * xi
        
        out += self.b
        out = out.tanh()

        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)

        return y
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    