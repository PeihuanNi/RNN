from RNN import *

class Encoder(nn.Module):
    def __init__(self, num_steps=28, input_size=28, hidden_size=128, num_classes=10):
        super(Encoder, self).__init__()
        self.num_steps = num_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.layers = []
        for _ in range(self.num_steps):
            self.layers.append(RNN(input_size, hidden_size, num_classes))
        self.layers = nn.ModuleList(self.layers)
        # self.layers = nn.ModuleList([RNN(input_size, hidden_size, num_classes) for _ in range(num_steps)])

    def forward(self, x):
        hidden = torch.zeros(1, self.hidden_size, device='cuda')
        for i in range(self.num_steps):
            hidden, output = self.layers[i](x[:, i, :], hidden)
        return output