import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import numpy as np
import dgl

class FakeNewsDataset(Dataset):

    def __init__(self, content, comment, comment_graph, labels, subgraphs , max_length_sentences=30, max_length_word=35):
        super(FakeNewsDataset, self).__init__()

        self.content = content
        self.comment = comment
        self.comment_graph = comment_graph
        self.labels = labels
        self.subgraphs= subgraphs

        self.max_length_sentences = max_length_sentences
        self 
        self.max_length_word = max_length_word

        self.num_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, samples):
        """Here, samples will be the sample returned by __getitem__ function"""
        content, comment, comment_graph ,label, subgraphs = map(list, zip(*samples))
        comment_graph= dgl.batch(comment_graph)
        comment = torch.from_numpy(np.array(comment))
        content = torch.from_numpy(np.array(content))
        label = torch.from_numpy(np.array(label))
        return content, comment, comment_graph, label, subgraphs


    def __getitem__(self, index):
        return self.content[index], self.comment[index], self.comment_graph[index], self.labels[index], self.subgraphs[index]
