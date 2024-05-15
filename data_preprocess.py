import os
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T
from torch.utils.data import TensorDataset

def read_imdb(path='./Imdb', is_train=True):
    reviews, labels = [], []
    tokenizer = get_tokenizer('basic_english') # 初始化英语分词器
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename), mode='r', encoding='utf-8') as f:
                reviews.append(tokenizer(f.read()))
                labels.append(1 if label == 'pos' else 0)
    return reviews, labels

def build_dataset(reviews, labels, vocab, max_len=512):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),   # 将文本转换成对应的词汇表索引
        T.Truncate(max_seq_len=max_len),  # 用于截断文本，将长度超过 max_len 的文本截断为 max_len 的长度
        T.ToTensor(padding_value=vocab['<pad>']),  # 文本转换成张量（tensor），'<pad>'为指定的填充值
        T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']), # 这个操作将文本进行填充，使得所有文本的长度都为 max_len。
    )
    dataset = TensorDataset(text_transform(reviews), torch.tensor(labels))
    return dataset


def load_imdb():
    reviews_train, labels_train = read_imdb(is_train=True)
    reviews_test, labels_test = read_imdb(is_train=False)
    vocab = build_vocab_from_iterator(reviews_train, min_freq=5, specials=['<pad>', '<unk>', '<cls>', '<sep>']) #构建词表 min_freq: 这个参数指定了词汇表中词语的最小频率
    vocab.set_default_index(vocab['<unk>']) # 设置词汇表的默认索引
    train_data = build_dataset(reviews_train, labels_train, vocab)
    test_data = build_dataset(reviews_test, labels_test, vocab)
    return train_data, test_data, vocab
