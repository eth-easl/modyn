class Buffer:

    def __init__(self, max_size=12):
        assert max_size%12 == 0, 'Currently only support buffers of sizes multiple of 12'

        self.bufferX = [None for _ in range(max_size)]
        self.bufferY = [None for _ in range(max_size)]
        self.weights = [None for _ in range(max_size)]
        self.size = 0
        self.max_size = max_size

    def replace(self, replace_this, replace_with):
        self.bufferX[replace_this] = self.bufferX[replace_with]
        self.bufferY[replace_this] = self.bufferY[replace_with]
        self.weights[replace_this] = self.weights[replace_with]
        self.bufferX[replace_with] = None
        self.bufferY[replace_with] = None
        self.weights[replace_with] = None

    def insert(self, index, x, y, weight=1):
        if self.bufferX[index] is None:
            raise Exception('Attempting to insert into empty slot. If buffer is not full (like now), please use insert_new')
        self.bufferX[index] = x
        self.bufferY[index] = y
        self.weights[index] = weight

    def pretty_labels(self):
        return str([a for a in self.bufferY])

    def get_size(self):
        return self.size
    
    def is_full(self):
        return self.size == self.max_size

    def insert_new(self, x, y, weight=1):
        if self.size >= self.max_size:
            raise Exception('Attempting to insert_new into a buffer that is already full. Use insert instead')
        else:
            self.bufferX[self.size] = x
            self.bufferY[self.size] = y
            self.weights[self.size] = weight
            self.size += 1
    