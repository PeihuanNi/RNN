import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.Wxh = nn.Linear(input_size, hidden_size, bias=False)
        self.bh = nn.Parameter(torch.randn([1, self.hidden_size]))
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bo = nn.Parameter(torch.randn([1, self.num_classes]))
        self.Who = nn.Linear(hidden_size, num_classes, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden):
        # print(f'***************************************************************************Type************************************************************************')
        # print(f'X \t{type(x)}')
        # print(f' former hidden \t{type(hidden)}')
        # print(f'Wxh \t{type(self.Wxh.weight.data)}')
        # print(f'Whh \t{type(self.Whh.weight.data)}')
        # print(f'Who \t{type(self.Who.weight.data)}')
        # print(f'bh \t{type(self.bh)}')
        # print(f'bo \t{type(self.bo)}')


        self.hidden = self.tanh(self.Wxh(x) + self.Whh(hidden) + self.bh)
        self.output = self.tanh(self.Who(self.hidden) + self.bo)
        # print(f'calc hidden \t{self.hidden.shape}')
        # print(f'output \t{self.output.shape}')
        # print('*********************************************************************************************************************************************************')
        return self.hidden, self.output
    
# x = torch.randn(28, 28, device='cuda')
# Net = RNN().to('cuda')
# hidden, output = Net(x[0], torch.zeros(1, 128, device='cuda'))
# print(hidden.shape)
# print(output.shape)