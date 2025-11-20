import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameter setting
batch_size = 100
num_classes = 10
hidden_size = 500
lr = 0.001
epochs = 5
# LeNet을 사용하기 위한 img size hyper parameter
image_size = 32

# CIFAR 데이터셋 전처리
# Normalize를 위한 데이터셋의 mean과 std 구하기
train_dataset = CIFAR10(root='./cifar', train=True, download=True)
mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0 # 각 축의 평균을 구하기 위해 axis=(0, 1, 2)를 사용
std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0 # 결과 형태 (3, ) -> r,g,b의 표준편차

trans = Compose([
    ToTensor(),
    Resize(image_size),
    Normalize(mean, std)
])

# 데이터셋 구성
train_dataset = CIFAR10(root='./cifar', train=True, transform=trans, download=True)
test_dataset = CIFAR10(root='./cifar', train=False, transform=trans, download=True)

# 데이터 로더
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 이미지 확인용
# trans 과정을 거친 후 이미지를 바로 확인하면 왜곡된 이미지가 나옴
def reverse_trans(x: torch.Tensor) -> torch.Tensor:
    """
    정규화된 픽셀의 값을 원래의 0-255 스케일로 역변환합니다.
    모델 학습을 위해 0-1 범위로 정규화 되었던 이미지를 시각화해 원본 픽셀 값으로 되돌립니다.

    Args:
        x(torch.Tensor): 정규화된 이미지 텐서 (float)
    
    Returns:
        torch.Tensor: 0-255 범위로 역변환된 픽셀 값을 갖는 텐서 (float).
                      (0과 1 사이로 클램핑된 후 255가 곱해짐)
    """
    x = (x * std) + mean
    return x.clamp(0, 1) * 255 

# 원본 이미지 반환 함수
def get_numpy_image(data):
    """
    PyTorch 텐서 이미지를 시각화를 위한 OpenCV (NumPy) 이미지 형태로 변환합니다.

    Args:
        data (torch.Tensor): PyTorch 이미지 텐서 (일반적으로 C, H, W 형태).

    Returns:
        np.ndarray: OpenCV 호환을 위해 BGR -> RGB로 변환된 H, W, C 형태의 NumPy 배열 이미지 (uint8).
    """
    # PyTorch의 (C, H, W) 형태를 OpenCV의 (H, W, C)로 변경하고 역정규화
    img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
    # OpenCV는 기본적으로 BGR 순서를 사용하므로, RGB 순서로 변환하여 반환
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# 이미지 복구
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1000

# train_dataset의 idx번째 데이터를 transform을 적용해서 가져옴.
img, label = train_dataset.__getitem__(idx)
img = cv2.resize(
    # 이미지를 numpy 형태로 변환
    get_numpy_image(img),
    # 512*512로 사이즈 변환
    (512, 512)
)
# 이미지에서 가져온 label은 0 ~ 9사이 숫자로 labels의 인덱스에 접근해 이미지에 맞는 label이 지정됨.
label = labels[label]

print(label)

# 기본 LeNet5
class MyLeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # b, c, h, w = x.shape
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(batch_size, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# nn.Sequential 기능 활용
class MyLeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 변수 명을 따로 선언할 필요가 없음
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features = 6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=6),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=16)
        )

        self.seq_fc = nn.Sequential(
            nn.Linear(in_features= 16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.seq_conv(x)
        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)
        x = self.seq_fc(x)

        return x
    
# linear 함수 추가
class MyLeNet_linear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.seq_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc_mid1 = nn.Linear(6*14*14, 2048)
        self.fc_mid2 = nn.Linear(2048, 6*14*14)

        self.seq_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.seq_fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(out_features=120, out_feautres=84),
            nn.Linear(in_features=84, out_features=num_classes)
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = self.seq_conv1(x)
        # conv layer를 통과해 변형된 b, w, h를 저장
        _, tmp_c, tmp_w, tmp_h = x.shape

        x = x.reshape(b, -1)
        x = self.fc_mid1(x)
        x = self.fc_mid2(x)
        # 다시 seq_conv1의 출력 사이즈와 동일하게 변형함
        x = x.reshape(b, tmp_c, tmp_h, tmp_w)
        x = self.seq_conv2(x)
        # 평탄화
        x = x.reshape(b, -1)
        x = self.seq_fc(x)

        return x
    
# conv2 layer 추가
class MyLeNet_conv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 크기 유지를 위해 kernel_size = 3
        # ModuleList 활용으로 반복되는 layer 효율적 관리
        self.add_conv1 = nn.ModuleList(
            [nn.Conv2d(3, 6, 3, 1, 1)] +
            [nn.Conv2d(6, 6, 3, 1, 1) for _ in range(2)]
        )

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        # add_conv1에서 출력된 (6, 14, 14) image를 받을 수 있는 conv layer 2개 생성
        # 마지막 layer는 conv_seq2의 입력으로 활용될 수 있게 16 차원으로 변경
        self.add_conv2 = nn.ModuleList([
            nn.Conv2d(6, 6, 3, 1, 1),   # (6, 14, 14)
            nn.Conv2d(6, 16, 3, 1, 1)   # (16, 14, 14)
        ])

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes)
        )
    
    def forward(self, x):
        # 반복문으로 module 안에 있는 리스트에서 1개씩 추출
        for module in self.add_conv1:
            x = module(x)
        x = self.conv_seq1(x)

        for module in self.add_conv2:
            x = module(x)
        x = self.conv_seq2(x)

        x = x.view(x.size(0), -1)
        x = self.fc_seq(x)
        return x
    
# conv 병합
class myLeNet5_incep(nn.Module):
    def __init__(self, num_clasees):
        super().__init__()
        self.conv_incep1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.conv_incep2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv_incep3 = nn.Conv2d(3, 6, 1, 1, 0)

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, stride=5, kernel_size=1),
            nn.BatchNorm2d(in_features = 6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_clasees)
        )

    def forward(self, x):
        x_1 = self.conv_incep1(x)
        x_2 = self.conv_incep2(x)
        x_3 = self.conv_incep3(x)
        x_cat = torch.cat((x_1, x_2, x_3), dim=1)

        x = self.conv_seq1(x_cat)
        x = self.conv_seq2(x_cat)

        b = x.shape[0]

        x = x.reshape(b, -1)
        x = self.fc_seq(x)

        return x
    
model = MyLeNet_conv(num_classes).to(device)
loss = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=lr)

def eval(model, dataloader):
    """
    model 평가 코드

    Args -> 앞서 선언한 모델, 예측 값과 실제 값을 불러오는 dataloader
    output -> 전체 개수 대비 맞춘 개수

    """
    correct = 0
    total = 0
    for img, trg in dataloader:
        img.to(device)
        trg.to(device)

        output = model(img)

        _, pred = torch.max(output, 1)

        correct += (pred == trg).sum().item()
        total += img.shape[0]
    return correct/total

step = 0
for epoch in range(epochs):
    for i, (image, target) in enumerate(train_dataloader):
        if step==522:
            # step에 중단점을 걸기위해 사용
            print(step)
        image.to(device)
        target.to(device)

        output = model(image)
        val_loss = loss(output, target)

        optim.zero_grad()
        val_loss.backward()
        optim.step()

        if i%100==0:
            print(val_loss.item())
            print('accuracy:', eval(model, test_dataloader))
        step += 1