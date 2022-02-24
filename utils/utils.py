import json
import logging

from collections import OrderedDict
from multiprocessing.sharedctypes import Value

ALL_DOMAINS = ['餐馆', '景点', '酒店', '出租', '地铁', "bye", 'greet', 'reqmore', 'thank', 'welcome']
All_ACTS = ['general', 'inform', 'nooffer', 'recommend', 'request', 'select']
DOMAINS_SLOTS = ['名称', '地址', '地铁', '电话', '营业时间', '推荐菜', '人均消费', '评分', '周边景点', '周边餐馆',
                '周边酒店', '门票', '游玩时间', '酒店类型', '酒店设施', '价格', '出发地', '目的地', '出发地附近地铁站',
                '目的地附近地铁站', '车型', '车牌', '出发地', '目的地']

def write_dict(fn, dic):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)

class Vocab(object):
    def __init__(self, vocab_size) -> None:
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0
        self._idx2word = {} # word + oov
        self._word2idx = {} # word
        self._freq_dict = {} # word + oov
        for w in ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>', '<eos_u>', 
        '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>', '<eos_d>']:
            self._add_to_vocab(w)

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: {:d}'.format(len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        
        for word in ALL_DOMAINS:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        
        for word in All_ACTS:
            word = '[' + word + ']'
            self._add_to_vocab(word)

        for word in DOMAINS_SLOTS:
            word = '[' + word + ']'
            self._add_to_vocab(word)
            temp_word = '[value_' + word + ']'
            self._add_to_vocab(temp_word)

        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path + '.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path + '.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('Vocab file loaded from ' + vocab_path)
        print('Vocabulary size including oov: {:d}'.format(self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)

    def encode(self, word, include_oov=False):
        if include_oov:
            if self._word2idx.get(word, None) is None:
                raise ValueError('Unknown word: {:s}. Vocabulary should include oovs here.'.format(word))
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        '''
        将oov的idx映射到<unk>
        '''
        return self._word2idx['<unk>'] if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]

    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: {:d}. Vocabulary should include oovs here.'.format(idx))
        if not indicate_oov or idx < self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx] + '(o)'

    def sentence_decode(self, index_list, eos=None, indicative_oov=False):
        l = [self.decode(_, indicative_oov) for _ in index_list]
        if not eos or eos not in l: # 移除掉eos
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]