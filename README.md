# Model Build

딥러닝 모델 구현 및 학습 코드 모음 (리팩토링 완료)

## 프로젝트 구조
```
model_build/
├── Transformer/              # Transformer 모듈 (모듈화 완료)
│   ├── misc/                 # 유틸리티 함수
│   │   └── tools.py
│   ├── modules/              # 데이터 처리 모듈
│   │   └── datasets.py
│   ├── networks/             # 모델 아키텍처
│   │   └── transformers_encoder.py
│   └── utils/                # 헬퍼 함수
│       ├── get_modules.py    # 모듈 동적 로드
│       ├── get_path.py       # 경로 관리
│       └── parser.py         # 인자 파싱
├── LeNet5_CIFAR10.py         # LeNet5 CIFAR-10 분류
├── mlp.py                    # MLP 구현
└── myLeNet5_tuning.py        # LeNet5 하이퍼파라미터 튜닝
```

## 모델 목록

### 1. Transformer (모듈화)

**구조:**
- `networks/`: Transformer Encoder 구현
- `modules/`: 데이터셋 로더
- `utils/`: 모듈 로드, 경로 관리, 인자 파싱
- `misc/`: 보조 함수

**사용 예시:**
```python
from Transformer.networks.transformers_encoder import TransformerEncoder
from Transformer.modules.datasets import load_dataset

model = TransformerEncoder(...)
data = load_dataset(...)
```

### 2. LeNet5

**파일:** `LeNet5_CIFAR10.py`

**구현 모델:**
- `MyLeNet5`: 기본 LeNet5
- `MyLeNet_linear`: FC layer 추가 버전
- `MyLeNet_conv`: Conv layer 추가 버전
- `myLeNet5_incep`: Inception 스타일 병합

**실행:**
```bash
python LeNet5_CIFAR10.py
```

**하이퍼파라미터:**
- Batch size: 100
- Epochs: 5
- Learning rate: 0.001
- Image size: 32x32

### 3. MLP

**파일:** `mlp.py`
- 기본 다층 퍼셉트론 구현

### 4. LeNet5 Tuning

**파일:** `myLeNet5_tuning.py`
- LeNet5 하이퍼파라미터 최적화 실험

## 요구사항
```bash
pip install torch torchvision opencv-python
```

## 특징

- 모듈화된 코드 구조로 재사용성 향상
- BatchNorm, ModuleList 등 PyTorch 최신 기능 활용
- CIFAR-10 자동 다운로드 및 전처리 파이프라인 포함
