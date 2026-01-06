import torch
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 차원의 크기는 Linear를 통과한 후 d_model을 그대로 유지해야 하니 개별적인 linear 생성
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        # Linear를 통과하고 나온 최종 출력값
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        b, seq_len, _ = q.shape

        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)
        # Q, K, V 크기 변환 : [batch_size, seq_len, d_model] ->  [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim)
        K = K.view(b, seq_len, self.num_heads, self.head_dim)
        V = V.view(b, seq_len, self.num_heads, self.head_dim)

        # attention score 계산
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # softmax를 사용해 확률값으로 변환
        attention_weights = torch.softmax(score, dim=-1) # [batch_size, seq_len, num_heads, num_heads]

        # attention socre를 V에 곱해 최종 결과값 연산
        attention = torch.matmul(attention_weights, V)

        # attention head 마다 계산된 결과를 하나로 연결
        attention = attention.reshape(b, seq_len, self.d_model)

        out = self.out(attention)

        return out

class TransformersLayer(nn.Module):
    """
    Encoder block에 들어가는 한 개의 Transformer layer 생성
    """
    def __init__(self, d_model, num_heads, feedforward_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # MultiHeadAttention(d_model, num_heads)의 결과 vector의 차원은 d_model이기 때문에 LayerNorm에 d_model 전달
        self.norm1 = nn.LayerNorm(d_model)
        self.feedfoward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # skip connection
        skip = x.clone()
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = self.norm1(x + skip)

        # feed forward
        skip = x.clone()
        x = self.feedfoward(x)
        x = self.dropout(x)
        x = self.norm2(x + skip)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq):
        super().__init__()
        # 위치 인코딩 행렬 생성
        self.position_encoding = torch.zeros(max_seq, d_model)
        # 0부터 max_seq까지의 값을 가지는 pos vector 생성. unsqueeze로 형태를 (max_seq, 1)로 만듦.
        pos = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        # 차원 및 분할 요소 계산
        dim = torch.arange(0, d_model, step=2, dtype=torch.float)
        div_term = torch.exp(-math.log(10000) * (dim / d_model))
        # 삼각 함수 계산
        self.position_encoding[:, 0::2] = torch.sin(pos * div_term)
        self.position_encoding[:, 1::2] = torch.cos(pos * div_term)

        # 모델 학습 방지를 위해 학습 변수가 아닌 상수로 저장
        self.register_buffer('positional_encoding', self.position_encoding.unsqueeze(0)) # 마지막 unsqueeze는 batch 차원 추가
    
    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1)]
        return x


class TransformersEncoder(nn.Module):
    def __init__(self, vocab_size, max_length, num_layer, d_model, num_heads, feedforward_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq=max_length)
        layers = []
        for _ in range(num_layer):
            layers.append(TransformersLayer(d_model, num_heads, feedforward_dim, dropout_rate))
        
        self.layers = nn.Sequential(*layers)

    def foward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.layers(x)
        return x

class IMDBClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = TransformersEncoder(args.vocab_size, args.max_length, args.num_layers, args.d_model, args.num_heads, args.feedforward_dim, args.dropout_rate)
        self.classifier = nn.Linear(args.d_model, args.num_classes)
    
    def forward(self, x):
        x = self.encoder(x) # [batch_size, seq_len, d_model]
        x = x[:, 0, :] # 가장 첫 위치에 있는 토큰의 임베딩만 사용
        x = self.classifier(x) # batch_size, num_classes

        return x