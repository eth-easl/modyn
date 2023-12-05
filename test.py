import torch, numpy, io, pickle
import torch.nn as nn
from pathlib import Path

TEST_DATA = Path.home() / "test_data"

def create_tiny_dataset():
    rng = numpy.random.default_rng()
    for n in range(1):
        a = rng.random(size=(2,2), dtype=numpy.float32)
        t = torch.from_numpy(numpy.array(a))
        torch.save(t, TEST_DATA / f"tensor_{n}.pt")
        label = torch.argmax(t, dim=1)
        torch.save(label, TEST_DATA / f"label_{n}.pt")

def load_tiny_dataset():
    for n in range(1):
        t = torch.load(TEST_DATA / f"tensor_{n}.pt")
        label = torch.load(TEST_DATA / f"label_{n}.pt")
        yield (t, label)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(2,2)

    def forward(self, x: torch.Tensor):
        x = self.output(x)
        return x
    

if __name__ == '__main__':
    torch.manual_seed(1)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    create_tiny_dataset()
    for t, label in load_tiny_dataset():
        opt.zero_grad()
        y = net(t)
        loss = criterion(y, label)
        loss.backward()
        # This was missing before
        opt.step()

