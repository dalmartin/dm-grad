class Value:

    def __init__(self, val, _children=[], _label=''):
        self.val = val
        self._prev = set(_children)
        self.grad = 0.0
        self.back = lambda: None
        self._label = _label

    def __repr__(self):
        return f"Value: {self.val}\tGradient: {self.grad}\n"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value((self.val + other.val), (self, other))

        def back():
            self.grad += 1.0 * res.grad
            other.grad += 1.0 * res.grad
            return
        res.back = back

        return res

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value((self.val * other.val), (self, other))
            
        def back():
            self.grad += other.val * res.grad
            other.grad += self.val * res.grad
        res.back = back

        return res

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        import math
        t = math.tanh(self.val)
        res = Value(t, (self,))

        def back():
            self.grad += (1 - t**2) * res.grad
        res.back = back

        return res

    def topo_sort(self, visited=None):
        sorted_nodes = []
        if visited is None:
            visited=set()

        if self not in visited:
            for node in self._prev:
                sorted_nodes.extend(node.topo_sort(visited))
                visited.add(node)

            visited.add(self)
            sorted_nodes.append(self)

        return sorted_nodes

    def backward(self):
        self.grad = 1.0

        # Topological sort (recursive sort of elements in order)
        sorted_nodes = self.topo_sort()

        #Then run .back() for every element in the sorted order
        sorted_nodes.reverse()
        for node in sorted_nodes:
            node.back()

if __name__ == '__main__':

    # Forward pass
    a = Value(8); a._label='a'
    b = Value(4); b._label='b'
    c = a + b; c._label='c'
    d = b * a; d._label='d'
    e = a + b; e._label='e' 
    f = d + e; f._label='f'
    R = c * f; R._label='R'
    R.grad = 1.0
    # Forward pass yields R value (Result)
    print(R)
    
    R.backward()
