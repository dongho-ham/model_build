from misc.tools import *
from torch.utils.data import DataLoader
from modules.datasets import IMDBDataset
from torch.nn.utils.rnn import pad_sequence

# dataset 생성
def get_dataset(args):
    train_dataset = IMDBDataset(args, split='train')
    test_dataset = IMDBDataset(args, vocab=train_dataset.vocab)
    return train_dataset, test_dataset

def collate_fn(batch):
    """
    데이터 배치를 정리하고, 패딩 추가
    """
    texts, lables = zip(*batch)

    # 텍스트 길이를 기준으로 정렬
    sorted_indices = sorted(range(len(texts)), key=lambda i: len(texts[i], reverse=True))
    # tensor로 변환
    texts = [torch.LongTensor(texts[i]) for i in sorted_indices]
    labels = [torch.LongTensor(labels[i]) for i in sorted_indices]

    # 패딩
    padded_texts = pad_sequence(texts, batch_first=True)

    labels = torch.cat(labels)

    return padded_texts, labels

# dataloader 생성
def get_dataloader(args):
    train_dataset, test_dataset = get_dataset(args)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    return train_loader, test_loader