from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import numpy as np
import torch
import dgl

class FakeNewsDataset(Dataset):

    def __init__(self, content, comment, comment_graph, labels, subgraphs, glove_path, max_length_sentences=30, max_length_word=35):
        super(FakeNewsDataset, self).__init__()

        self.content = self._encode_texts(content)
        self.comment = self._encode_texts(comment)
        self.comment_graph = comment_graph
        self.labels = labels
        self.subgraphs= subgraphs
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word

        self.num_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)
    
    def _encode_texts(self, texts):
        """
        Pre process the news content sentences to equal length for feeding to GRU
        :param texts:
        :return:
        """
        encoded_texts = np.zeros((len(texts), self.max_sents, self.max_sen_len), dtype='int32')
        for i, text in enumerate(texts):
            # ids = [item.ids for item in self.tokenizer.encode_batch(text)]
            ids = self.tokenizer(text, return_tensors='np', padding=True, truncation=True, max_length=self.max_sen_len)['input_ids']
            encoded_text = np.array(
                torch.nn.functional.pad(
                    torch.from_numpy(ids),
                    pad=(0, self.max_sen_len - torch.tensor(ids).shape[1]), 
                    mode='constant', 
                    value=0
                )
            )[:self.max_sents]
            
            encoded_texts[i][:len(encoded_text)] = encoded_text
        return encoded_texts

    def collate_fn(self, samples):
        """Here, samples will be the sample returned by __getitem__ function"""
        content, comment, comment_graph ,label, subgraphs = map(list, zip(*samples))
        comment_graph= dgl.batch(comment_graph)
        comment = torch.from_numpy(np.array(comment))
        content = torch.from_numpy(np.array(content))
        label = torch.from_numpy(np.array(label))
        return content, comment_graph, label, subgraphs

    def __getitem__(self, index):
        return self.content[index], self.comment[index], self.comment_graph[index], self.labels[index], self.subgraphs[index]