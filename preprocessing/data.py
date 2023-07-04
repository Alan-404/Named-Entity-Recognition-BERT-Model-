import io
import json
import os
import pickle
import numpy as np
from glob import glob

class DataProcessor:
    def __init__(self, tokenizer_path: str = None, init_tokens: list = ["<pad>", "<start>", "<end>"], oov: bool = True) -> None:
        self.dictionary = []
        self.entities = []

        if tokenizer_path is None or os.path.exists(tokenizer_path) == False:
            for token in init_tokens:
                self.add_token(token)
                self.add_entity(token)
            if oov:
                self.add_token("<oov>")
        else:
            self.load_tokenizer(tokenizer_path)

        self.tokenizer_path = tokenizer_path

    def add_token(self, token: str):
        self.dictionary.append(token)

    def add_entity(self, entity: str):
        self.entities.append(entity)


    def load_tokenizer(self, path:str):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        self.dictionary = data['dictionary']
        self.entities = data['entity']

    def save_tokenizer(self, path: str):
        data = {
            'dictionary': self.dictionary,
            'entity': self.entities
        }
        with open(path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def fit(self, texts: list, tags: list = None, train: bool = True, start_token: bool = True, end_token: bool = True):
        if train == True and tags is None:
            return
        digits = []
        entities = []
        if start_token:
            digits.append(self.dictionary.index("<start>"))
            entities.append(self.entities.index("<start>"))
        for index, text in enumerate(texts):
            if text not in self.dictionary:
                if train == True:
                    self.add_token(text)
                    digits.append(len(self.dictionary) - 1)
                else:
                    digits.append(self.dictionary.index("<oov>"))
            else:
                digits.append(self.dictionary.index(text))
            
            if train == True:
                if tags[index] not in self.entities:
                    self.add_entity(tags[index])
                    entities.append(len(self.entities) - 1)
                else:
                    entities.append(self.entities.index(tags[index]))
        if end_token:
            digits.append(self.dictionary.index("<end>"))
            entities.append(self.entities.index("<end>"))
        return np.array(digits), np.array(entities)
    
    def padding_sequence(self, sequence, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)    

    def save_data(self, data: np.ndarray, path: str):
        with open(path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)    
    
    def load_data(self, path: str):
        with open(path + "/word.pkl", 'rb') as file:
            word = pickle.load(file)
        with open(path + "/tag.pkl", 'rb') as file:
            tag = pickle.load(file)

        return word, tag

    def process(self, file_path: str, max_length: int = None, data_path: str = None):
        files = glob(f"{file_path}/*.json")
        dataset = []
        for file in files:
            dataset += io.open(file, encoding='utf-8').read().strip().split("\n")
        word_digits = []
        maxlen = 0
        entity_digits = []
        for line in dataset:
            data = json.loads(line)
            word_digit, entity_digit = self.fit(data['words'], data['tags'])
            word_digits.append(word_digit)
            entity_digits.append(entity_digit)

            if maxlen < len(word_digit):
                maxlen = len(word_digit)

        if max_length is not None:
            maxlen = max_length

        word_digital = self.pad_sequences(word_digits, maxlen)
        entity_digital = self.pad_sequences(entity_digits, maxlen)

        if data_path is not None:
            self.save_data(word_digital, data_path + "/word.pkl")
            self.save_data(entity_digital, data_path + "/tag.pkl")

        if self.tokenizer_path:
            self.save_tokenizer(self.tokenizer_path)
        else:
            self.save_tokenizer("./tokenizer.pkl")

        return word_digital, entity_digital



