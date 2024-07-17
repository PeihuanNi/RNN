import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(input_size, hidden_size, bias=False)  # 输入到隐状态
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)  # 隐状态到隐状态
        self.Who = nn.Linear(hidden_size, num_classes, bias=False)  # 隐状态到输出
        self.tanh = nn.Tanh()
        self.bh = nn.Parameter(torch.randn(1, hidden_size))
        self.bo = nn.Parameter(torch.rand(1, num_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        # print(x.shape)
        hidden = self.tanh(self.Wxh(x) + self.Whh(hidden) + self.bh)  # 更新隐状态
        output = self.Who(hidden) + self.bo  # 生成输出
        # print(output.shape)
        return hidden, output

class RNNModel(nn.Module):
    def __init__(self, num_steps=28, input_size=28, hidden_size=128, num_classes=10):
        super(RNNModel, self).__init__()
        self.num_steps = num_steps
        self.rnn = RNN(input_size, hidden_size, num_classes)

    def forward(self, x):
        print(x.shape)
        hidden = torch.zeros(x.size(0), self.rnn.hidden_size, device=x.device)  # 初始化隐状态
        outputs = []
        
        for i in range(self.num_steps):
            hidden, output = self.rnn(x[:, i, :], hidden)  # 更新隐状态并生成输出
            outputs.append(output)

        return outputs[-1]  # 返回最后一个时间步的输出

# 超参数
input_size = 28
sequence_length = 28
batch_size = 64
hidden_size = 128
num_classes = 10
num_epochs = 10
learning_rate = 1e-4

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='D:/Dataset', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='D:/Dataset', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数和优化器
model = RNNModel(sequence_length, input_size, hidden_size, num_classes).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, sequence_length, input_size).to('cuda')  # 调整形状
        labels = labels.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试过程
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, sequence_length, input_size).to('cuda')  # 调整形状
        print(images.shape)
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
