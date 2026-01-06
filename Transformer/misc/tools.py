import torch
import numpy as np

def evaluation(model, test_loader, args):
    """
    전체 정확도 평가
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (text, label) in enumerate(test_loader):
            text = text.to(args.device)
            label = label.to(args.device)

            output = model(text)
            _, pred = torch.max(output, 1) # 긍부정 확률 값 중 큰 값만 사용
            correct += (pred == label).sum().item() # True만 count
            total += len(label)
    model.train() # 다음 epoch 시작 전에 학습 모드로 변환 
    return correct / total

def eval_by_class(model, test_loader, args):
    """
    클래스 별 정확도 평가 (긍정/부정)
    """
    model.eval()
    correct = np.zeros(args.num_classes)
    total = np.zeros(args.num_classes)

    with torch.no_grad():
        for _, (text, label) in enumerate(test_loader):
            text = text.to(args.device)
            label = text.to(args.device)

            output = model(text)
            _, pred = torch.max(output, 1)
            for i in range(args.num_classes):
                # 예측과 정답이 같아야 1*1로 1씩 count
                correct[i] += ((pred == i) * (label == i)).sum().item()
            model.train()
    return correct / total, sum(correct) / sum(total)
