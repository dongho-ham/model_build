import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼 파라미터 세팅
batch_size = 100
num_classes = 10
lr = 0.001
epochs = 5
image_size = 32

trans = Compose([
    Resize(image_size),
    ToTensor()
])

train_dataset = datasets.MNIST('./mnist', transform=trans, train=True, download=True)
test_dataset = datasets.MNIST('./mnist', transform=trans, train=False, download=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class myLeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        
        self.fc1 = nn.Linear(6*14*14, 2048)
        self.fc2 = nn.Linear(2048, 6*14*14)
        self.fc3 = nn.Linear(16*5*5, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        x = x.reshape(batch_size, 6, 14, 14)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(batch_size, -1)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x
    
model = myLeNet5(num_classes).to(device)
optim = Adam(model.parameters(), lr=lr)
criteria = nn.CrossEntropyLoss()

losses = []

for epoch in range(epochs):
    for idx, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criteria(output, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % 100 == 0:
            print(f'{epoch}/{epochs}, {idx} Step | Loss: {loss.item():.4f}')
            losses.append(loss.item())
