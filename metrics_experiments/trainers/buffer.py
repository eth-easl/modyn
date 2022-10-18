class Buffer:

    def __init__(self, max_size=12):
        assert max_size%12 == 0, 'Currently only support buffers of sizes multiple of 12'

        self.bufferX = [None for _ in range(max_size)]
        self.bufferY = [None for _ in range(max_size)]
        self.weights = [None for _ in range(max_size)]


    def replace(self, replace_this, replace_with):
        self.bufferX[replace_this] = self.bufferX[replace_with]
        self.bufferY[replace_this] = self.bufferY[replace_with]
        self.weights[replace_this] = self.weights[replace_with]

    def insert(self, index, x, y, weight=1):
        self.bufferX[index] = x
        self.bufferY[index] = y
        self.weights[index] = weight

    def pretty_labels(self):
        return str([a.item() for a in self.bufferY])