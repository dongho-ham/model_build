import re
import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

def preprocess_text(text):
    """
    텍스트 데이터셋을 정제합니다.

    Args: 문장

    Output: 정제 문장
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) # 특수문자 제거 
    text = re.sub(r"<.*?>", "", text) # HTML태그 제거 
    text = re.sub(r"\s+", " ", text) # 연속된 띄어쓰기 제거
    text = text.strip() # 문장 앞뒤 불필요한 공백 제거
    return text 

# IMDB 데이터셋 생성
class IMDBDataset(Dataset):
    """
    IMDB 데이터셋을 불러오고 전처리 한 뒤 토큰화합니다.
    
    Args: max_length, split, vocab

    Output: 토큰화된 문장
    """
    def __init__(self, args, split='train', vocab=None):
        # 데이터 불러오기
        raw_dataset = load_dataset('imdb', split=split)

        # 영어 문장을 단어 단위로 분리
        self.tokenizer = get_tokenizer('basic_english')
        
        self.texts = [] # 전처리된 원본 텍스트
        self.token_texts = [] # 토큰화된 텍스트 (단어 리스트)
        self.all_tokens = [] # 모든 토큰이 저장된 리스트

        prog = tqdm.tqdm(raw_dataset['text'])
        
        for line in prog:
            prog.set_description(f"Processing {split} dataset")
            proc_line = preprocess_text(line)
            self.texts.append(proc_line)

            # 영어 문장 토큰화
            tokend_line = self.tokenizer(proc_line)
            # 최대 길이만큼 자르기
            tokend_line = tokend_line[:args.max_length]
            # 토큰화된 문장 저장
            self.token_texts.append(tokend_line)
            # vocabulary 만들기 위해 사용
            self.all_tokens += tokend_line
        
        # Vocabulary 생성 후 단어 <-> 숫자 변환
        self.all_tokens.append("<unk>") # unknown 단어 처리
        self.all_tokens.append("<pad>") # padding 단어 처리
        self.labels = raw_dataset['label'] # 긍정, 부정 레이블 저장

        if split == 'train':
            self.vocab = ['<pad>']
            self.vocab += list(set(self.all_tokens)) # 중복 제거
        else:
            self.vocab = vocab
        
        self.vocab_size = len(self.vocab)
        args.vocab_size = self.vocab_size
        # 단어 <-> 숫자 변환
        self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx:word for idx, word in enumerate(self.vocab)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        특정 인덱스의 데이터를 1개 가져옵니다.

        Args: 인덱스

        Output: 인덱싱 문장, 레이블
        예시: [2, 3, 4, 5], 1
        """
        tokened_line = self.token_texts[idx]
        # 아는 단어면 단어를 가져오고, 모르면 unk를 가져옴.
        idxed_line = [self.word2idx.get(word, self.word2idx['<unk>']) for word in tokened_line]
        label = self.labels[idx]

        return idxed_line, label