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


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.upsample = nn.Linear(num_classes, 100)
        self.downsample = nn.Linear(100, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.downsample(x)
        # output = self.softmax(x)
        return x


class Net(nn.Module):
    def __init__(self, num_steps, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.num_steps = num_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # self.rnn = RNN(self.input_size, self.hidden_size, self.num_classes)
        self.encoder = Encoder(self.num_steps, self.input_size, self.hidden_size, self.num_classes)
        self.decoder = Decoder(self.num_classes)


    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output
    
input_size = 28
sequance_length = 28
batch_size = 64
hidden_size = 128
time_step = 28
num_epochs = 5
num_classes = 10
lr = 0.1
RNN_MNIST = Net(time_step, input_size, hidden_size, num_classes).to('cuda')
# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='D:/Dataset', 
                                           train=True, 
                                           transform=transform, 
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='D:/Dataset', 
                                          train=False, 
                                          transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(RNN_MNIST.parameters(), lr=lr)

n_epochs = 15

RNN_MNIST.train()
picture = 0
for epoch in range(n_epochs):
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        images = images.view(-1, 28, 28) # 这里一定要用-1，否则在不满一个batch size的时候就会报错
        # print(images.shape)
        optimizer.zero_grad()
        output = RNN_MNIST(images)
        # print(output.shape)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')

RNN_MNIST.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        images = images.view(-1, 28, 28)
        # print(images.shape)
        outputs = RNN_MNIST(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

torch.save(RNN_MNIST, 'RNN_MNIST.pth')
