import os
import time
import gc
import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import dgl
from hyptorch.geoopt.optim.radam import RiemannianAdam
from model.utils.metrics import Metrics
from model.Hyphen.poincare import Hyphen as PoincareHyphen
from model.Hyphen.euclidean import Hyphen as EuclidHyphen
from model.ssm.hs4 import SSM4RC 
from model.bert.bert import HBert 
from model.han.han import Han
from model.utils.dataset import FakeNewsDataset
from model.utils.utils import get_evaluation
import wandb
from transformers import AutoTokenizer
from const import * 
from hyptorch.geoopt import PoincareBall, Euclidean
from hyptorch.lorentz.manifold import CustomLorentz 
from accelerate import Accelerator
import pathlib
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union

from hyptorch.geoopt.manifolds.lorentz.math import lorentz_to_poincare, poincare_to_lorentz

import umap


accelerator = Accelerator()
class Trainer:
    def __init__(
        self,
        manifold,
        model_type,
        platform,
        max_sen_len,
        max_com_len,
        max_sents,
        max_coms,
        lr,
        fourier,
        curv=1.0,
        enable_log=False,
        embedding_dim=100,
    ):
        self.lr = lr
        self.model_type = model_type
        self.max_sents = max_sents
        self.max_coms = max_coms
        self.max_sen_len = max_sen_len
        self.max_com_len = max_com_len
        self.metrics = Metrics()
        self.fourier = fourier
        self.platform = platform
        self.embedding_dim = embedding_dim 
        self.device = accelerator.device 
        self.manifold_name = manifold

        if manifold == EUCLID:
            self.manifold = Euclidean()
        elif manifold == POINCARE:
            self.manifold = PoincareBall(c=curv)
        else:
            self.manifold = CustomLorentz(k=curv)
     
        self.log_enable = enable_log 

        print('using manifold ',  manifold)
        print('using fourier',  fourier)

    def log(self, stats):
        if self.log_enable:
            wandb.log(stats)
            

    def _fit_on_texts(self):
        """
        Creates vocabulary set from the news content and the comments
        """
     
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print("saved tokenizer")



    def _build_hyphen(self, n_classes=2, batch_size=12):
        """
        This function is used to build Hyphen model.
        """
        embeddings_index = {}

        self.glove_dir = f"{ROOT_PATH}/glove.twitter.27B.{self.embedding_dim}d.txt"
        # self.glove_dir = f"{ROOT_PATH}/poincare_glove_100D_cosh-dist-sq_init_trick.txt"

        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            

        f.close()

        # get word index
        word_index = self.tokenizer.vocab
        embedding_matrix = np.random.random((len(word_index) + 1, self.embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        (
            self.word_hidden_size,
            self.sent_hidden_size,
            self.graph_hidden,
        ) = (self.embedding_dim//2, self.embedding_dim//2, self.embedding_dim)
        if isinstance(self.manifold, PoincareBall):
            model = PoincareHyphen(
                manifold=self.manifold,
                embedding_matrix=embedding_matrix,
                embedding_dim=self.embedding_dim,
                latent_dim=self.embedding_dim,
                word_hidden_size=self.word_hidden_size,
                sent_hidden_size=self.sent_hidden_size,
                device=self.device,
                graph_hidden=self.graph_hidden,
                batch_size=batch_size,
                num_classes=n_classes,
                max_comment_count=self.max_coms,
                fourier=self.fourier,
            )
        elif isinstance(self.manifold, Euclidean):
            model = EuclidHyphen(
                manifold=self.manifold,
                embedding_matrix=embedding_matrix,
                embedding_dim=self.embedding_dim,
                latent_dim=self.embedding_dim,
                word_hidden_size=self.word_hidden_size,
                sent_hidden_size=self.sent_hidden_size,
                device=self.device,
                graph_hidden=self.graph_hidden,
                batch_size=batch_size,
                num_classes=n_classes,
                max_comment_count=self.max_coms,
                fourier=self.fourier,
            )
        else: 
            raise RuntimeError("Does not support lorentz")
            
        print(f"{self.model_type} built")
        self.model = model

    def _build_ssm4rc(self, n_classes=2, batch_size=12):
        """
        This function is used to build Hyphen model.
        """
        embeddings_index = {}

        self.glove_dir = f"{ROOT_PATH}/glove.twitter.27B.{self.embedding_dim}d.txt"
        # self.glove_dir = f"{ROOT_PATH}/poincare_glove_100D_cosh-dist-sq_init_trick.txt"

        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            

        f.close()

        # get word index
        word_index = self.tokenizer.vocab
        embedding_matrix = np.random.random((len(word_index) + 1, self.embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        (
            self.word_hidden_size,
            self.sent_hidden_size,
            self.graph_hidden,
        ) = (self.embedding_dim//2, self.embedding_dim//2, self.embedding_dim)

        model = SSM4RC(
            manifold=self.manifold,
            embedding_dim=self.embedding_dim,
            latent_dim=self.embedding_dim,
            embedding_matrix=embedding_matrix,
            word_hidden_size=self.word_hidden_size,
            sent_hidden_size=self.sent_hidden_size,
            device=self.device,
            graph_hidden=self.graph_hidden,
            batch_size=batch_size,
            num_classes=n_classes,
            fourier=self.fourier,
        )

        print(f"{self.model_type} built")
        self.model = model


    def _build_han(self, n_classes=2, batch_size=12):
        embeddings_index = {}

        self.glove_dir = f"{ROOT_PATH}/glove.twitter.27B.{self.embedding_dim}d.txt"
        # self.glove_dir = f"{ROOT_PATH}/poincare_glove_100D_cosh-dist-sq_init_trick.txt"

        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            

        f.close()

        # get word index
        word_index = self.tokenizer.vocab
        embedding_matrix = np.random.random((len(word_index) + 1, self.embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        (
            self.word_hidden_size,
            self.sent_hidden_size,
            self.graph_hidden,
        ) = (self.embedding_dim//2, self.embedding_dim//2, self.embedding_dim)

        model = Han(
            manifold=self.manifold,
            embedding_dim=self.embedding_dim,
            latent_dim=self.embedding_dim,
            embedding_matrix=embedding_matrix,
            word_hidden_size=self.word_hidden_size,
            sent_hidden_size=self.sent_hidden_size,
            device=self.device,
            graph_hidden=self.graph_hidden,
            batch_size=batch_size,
            num_classes=n_classes,
            fourier=self.fourier,
        )

        print(f"{self.model_type} built")
        self.model = model


    def _build_bert(self, n_classes=2, batch_size=12):
        embeddings_index = {}

        self.glove_dir = f"{ROOT_PATH}/glove.twitter.27B.{self.embedding_dim}d.txt"
        # self.glove_dir = f"{ROOT_PATH}/poincare_glove_100D_cosh-dist-sq_init_trick.txt"

        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            

        f.close()

        # get word index
        word_index = self.tokenizer.vocab
        embedding_matrix = np.random.random((len(word_index) + 1, self.embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        (
            self.word_hidden_size,
            self.sent_hidden_size,
            self.graph_hidden,
        ) = (self.embedding_dim//2, self.embedding_dim//2, self.embedding_dim)

        model = HBert(
            manifold=self.manifold,
            embedding_dim=self.embedding_dim,
            latent_dim=self.embedding_dim,
            embedding_matrix=embedding_matrix,
            word_hidden_size=self.word_hidden_size,
            sent_hidden_size=self.sent_hidden_size,
            device=self.device,
            graph_hidden=self.graph_hidden,
            batch_size=batch_size,
            num_classes=n_classes,
            fourier=self.fourier,
        )

        print(f"{self.model_type} built")
        self.model = model

        
    def _encode_texts(self, texts, max_sents, max_sen_len):
        """
        Pre process the news content sentences to equal length for feeding to GRU
        :param texts:
        :return:
        """
        encoded_texts = np.zeros((len(texts), max_sents, max_sen_len), dtype='int32')
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
            )[:max_sents]
            
            encoded_texts[i][:len(encoded_text)] = encoded_text
        return encoded_texts

        
    def save_results(self, f1, prec, rec, acc,train_time, eval_time):
        file_name = f'{self.platform}.csv'
        path = f'{os.getcwd()}/results/{file_name}'
        if not pathlib.Path(path).is_file():
            head = "model,manifold,fourier,max_coms,max_sents,max_sen_len,max_com_len,embedding_dim,f1,prec,rec,acc,train time,eval time\n"
            with open(path, "a") as myfile:
                myfile.write(head)
        row = f'{self.model_type},{self.manifold_name},{self.fourier},{self.max_coms},{self.max_sents},{self.max_sen_len},{self.max_com_len},{self.embedding_dim},{f1},{prec},{rec},{acc},{train_time},{eval_time}\n'
        with open(path, "a") as myfile:
            myfile.write(row)


    def run(
        self,
        train_x,
        train_raw_c,
        train_y,
        train_c,
        val_c,
        val_raw_c,
        val_y,
        val_x,
        sub_train,
        sub_val,
        batch_size=9,
        epochs=5,
        eval=False,
        visualize=False,
    ):

        # Fit the vocabulary set on the content and comments
        self._fit_on_texts()

        print("Building model....")
        if self.model_type == HYPHEN:
            self._build_hyphen(n_classes=train_y.shape[-1], batch_size=batch_size)
        elif self.model_type == BERT:
            self._build_bert(n_classes=train_y.shape[-1], batch_size=batch_size)
        elif self.model_type == HAN:
            self._build_han(n_classes=train_y.shape[-1], batch_size=batch_size)
        else:
            self._build_ssm4rc(n_classes=train_y.shape[-1], batch_size=batch_size)

        self.model = accelerator.prepare(self.model) 
        self.optimizer = RiemannianAdam(self.model.parameters(), lr=self.lr)
        self.optimizer = accelerator.prepare(self.optimizer)
        if self.log_enable:
            wandb.init(
                project=self.platform,
                name=f'{self.model_type}_{self.embedding_dim}_{self.manifold_name}_{self.fourier}',
                config={
                    'type': self.model_type,
                    'manifold': self.manifold_name,
                    'embedding_dim': self.embedding_dim,
                    'max_sents': self.max_sents,
                    'max_coms': self.max_coms,
                    'fourier': self.fourier
                }
            )

        self.criterion = nn.CrossEntropyLoss()

        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x, self.max_sents,self.max_sen_len)
        encoded_val_x = self._encode_texts(val_x, self.max_sents,self.max_sen_len)
        encoded_train_c = self._encode_texts(train_raw_c, self.max_coms, self.max_sen_len)
        encoded_val_c = self._encode_texts(val_raw_c, self.max_coms, self.max_sen_len)
        print("preparing dataset....")

        # adding self loops in the dgl graphs
        train_c = [dgl.add_self_loop(i) for i in train_c]
        val_c = [dgl.add_self_loop(i) for i in val_c]
        train_dataset = FakeNewsDataset(
            content=encoded_train_x,
            comment=encoded_train_c,
            comment_graph=train_c,
            labels=train_y,
            subgraphs=sub_train,
        )
        val_dataset = FakeNewsDataset(
            content=encoded_val_x,
            comment=encoded_val_c,
            comment_graph=val_c,
            labels=val_y,
            subgraphs=sub_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        self.dataset_sizes = {
            "train": train_dataset.__len__(),
            "val": val_dataset.__len__(),
        }
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader) 
        self.dataloaders = {"train": train_loader, "val": val_loader}
        print("Dataset prepared.")
        if eval:
            self.load_model()
            return self.evaluate()
        elif visualize:
            self.load_model()
            return self.visualize_embeddings()
        return self.run_epoch(epochs)
        
    def run_epoch(self, epochs):
        """
        Function to train model for given epochs
        """

        start = time.time()

        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_acc = 0.0

        for epoch in range(epochs):
            accelerator.clear()
            accelerator.free_memory()
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 100)
            self.metrics.on_train_begin()
            self.model.train()

            num_iter_per_epoch = len(self.dataloaders["train"])
            for iter, sample in enumerate(tqdm.tqdm(self.dataloaders["train"])):
                self.optimizer.zero_grad()

                content, comment, comment_graph ,label, subgraphs = sample
                if self.model_type == HYPHEN:
                    self.model.content_encoder._init_hidden_state(len(label))
                    predictions,_,_ = self.model(
                        content=content, comment=comment_graph, subgraphs=subgraphs
                    )
                else:
                    predictions,_,_ = self.model(
                        content=content, comment=comment
                    )  
                loss = self.criterion(predictions, label)
                accelerator.backward(loss)
                self.optimizer.step()

                training_metrics = get_evaluation(
                    torch.max(label, 1)[1].cpu().numpy(),
                    predictions.cpu().detach().numpy(),
                    list_metrics=["accuracy"],
                )
                self.log({
                    "Train/Loss": loss, 
                    "Epoch": epoch * num_iter_per_epoch + iter,
                    "Train/Accuracy":training_metrics["accuracy"],
                })

            train_time = time.time() - start
            start = time.time()
            self.model.eval()
            loss_ls = []
            total_samples = 0
            for i, sample in enumerate(self.dataloaders["val"]):
                content, comment, comment_graph,label, subgraphs = sample
                num_sample = len(label)  # last batch size
                total_samples += num_sample
                if self.model_type == HYPHEN:
                    self.model.content_encoder._init_hidden_state(num_sample)
                    predictions,_,_ = self.model(
                        content=content, comment=comment_graph, subgraphs=subgraphs
                    ) 
                else:
                    predictions,_,_ = self.model(
                        content=content, comment=comment 
                    )  # As and Ac are the attention weights we are returning
                te_loss = self.criterion(predictions, label)
                loss_ls.append(te_loss * num_sample)
                _, predictions = torch.max(torch.softmax(predictions, dim=-1), 1)
                _, label = torch.max(label, 1)
                predictions = predictions.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                self.metrics.on_batch_end(epoch, i, predictions, label)

            acc, f1, prec, rec = self.metrics.on_epoch_end(epoch)
            eval_time= time.time() - start
            if f1 > best_f1:
                print(f"Best F1: {f1}")
                print("Saving best model!")
                self.log({'epoch':epoch, 'best F1': f1, 'best precision': prec, 'best recall': rec})
                dst_dir = f"saved_models/{self.platform}/{self.model_type}"
                os.makedirs(dst_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(), f"{dst_dir}/best_model_{self.manifold_name}.pt"
                )
                self.best_model = self.model
                best_f1 = f1
                best_precision = prec 
                best_recall = rec 
                best_acc = acc 

            te_loss = sum(loss_ls) / total_samples
            self.log({
                "Test/epoch": epoch,
                "Test/Loss": te_loss,
                "Test/Accuracy": acc,
                "Test/F1": f1,
                
            })
        print(f"Best F1: {best_f1}")
        self.save_results(best_f1, best_recall, best_precision, best_acc, train_time, eval_time)
        print("Training  end")
        print("-" * 100)



    def evaluate(self):
        self.model.eval()
        loss_ls = []
        total_samples = 0
        self.metrics.on_eval_begin()
        for i, sample in enumerate(self.dataloaders["val"]):
            content, comment, comment_graph,label, subgraphs = sample
            num_sample = len(label)  # last batch size
            total_samples += num_sample
            if self.model_type == HYPHEN:
                self.model.content_encoder._init_hidden_state(num_sample)
                predictions,_,_ = self.model(
                    content=content, comment=comment_graph, subgraphs=subgraphs
                ) 
            else:
                predictions,_,_ = self.model(
                    content=content, comment=comment 
                )  # As and Ac are the attention weights we are returning
            te_loss = self.criterion(predictions, label)
            loss_ls.append(te_loss * num_sample)
            _, predictions = torch.max(torch.softmax(predictions, dim=-1), 1)
            _, label = torch.max(label, 1)
            predictions = predictions.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            self.metrics.on_batch_end(100, i, predictions, label)

        return self.metrics.on_epoch_end(100)
 
    def process_atten_weight(
        self, encoded_text, content_word_level_attentions, sentence_co_attention
    ):
        """
        Process attention weights for sentence
        """
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.tokenizer.decode(wd_idx)
                    no_pad_sen_att.append((wd, content_word_level_attentions[k][i][j]))

                tmp_no_pad_text_att.append(
                    (no_pad_sen_att, sentence_co_attention[k][i])
                )
            no_pad_text_att.append(tmp_no_pad_text_att)

        # Normalize without padding tokens
        no_pad_text_att_normalize = None
        for npta in no_pad_text_att:
            if len(npta) == 0:
                continue
            sen_att, sen_weight = list(zip(*npta))
            new_sen_weight = [float(i) / sum(sen_weight) for i in sen_weight]
            new_sen_att = []
            for sw in sen_att:
                word_list, att_list = list(zip(*sw))
                att_list = [float(i) / sum(att_list) for i in att_list]
                new_wd_att = list(zip(word_list, att_list))
                new_sen_att.append(new_wd_att)
            no_pad_text_att_normalize = list(zip(new_sen_att, new_sen_weight))

        return no_pad_text_att_normalize

    def process_atten_weight_com(self, encoded_text, sentence_co_attention):
        """
        Process attention weight for comments
        """

        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.tokenizer.decode[wd_idx]
                    no_pad_sen_att.append(wd)
                tmp_no_pad_text_att.append(
                    (no_pad_sen_att, sentence_co_attention[k][i])
                )

            no_pad_text_att.append(tmp_no_pad_text_att)

        return no_pad_text_att

    def benchmark( self,
        train_x,
        train_raw_c,
        train_y,
        train_c,
        val_c,
        val_raw_c,
        val_y,
        val_x,
        sub_train,
        sub_val,
        batch_size=32,
    ):
        self._fit_on_texts()

        print("Building model....")
        if self.model_type == HYPHEN:
            self._build_hyphen(n_classes=train_y.shape[-1], batch_size=batch_size)
        elif self.model_type == BERT:
            self._build_bert(n_classes=train_y.shape[-1], batch_size=batch_size)
        elif self.model_type == HAN:
            self._build_han(n_classes=train_y.shape[-1], batch_size=batch_size)
        else:
            self._build_ssm4rc(n_classes=train_y.shape[-1], batch_size=batch_size)

        self.model = accelerator.prepare(self.model) 
        self.model.eval()
        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x, self.max_sents,self.max_sen_len)
        encoded_val_x = self._encode_texts(val_x, self.max_sents,self.max_sen_len)
        encoded_train_c = self._encode_texts(train_raw_c, self.max_coms, self.max_sen_len)
        encoded_val_c = self._encode_texts(val_raw_c, self.max_coms, self.max_sen_len)
        print("preparing dataset....")

        # adding self loops in the dgl graphs
        train_c = [dgl.add_self_loop(i) for i in train_c]
        val_c = [dgl.add_self_loop(i) for i in val_c]
        train_dataset = FakeNewsDataset(
            content=encoded_train_x,
            comment=encoded_train_c,
            comment_graph=train_c,
            labels=train_y,
            subgraphs=sub_train,
        )
        val_dataset = FakeNewsDataset(
            content=encoded_val_x,
            comment=encoded_val_c,
            comment_graph=val_c,
            labels=val_y,
            subgraphs=sub_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        self.dataset_sizes = {
            "train": train_dataset.__len__(),
            "val": val_dataset.__len__(),
        }
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader) 
        self.dataloaders = {"train": train_loader, "val": val_loader}

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("model_inference"):
                for i in range(10):
                    content, comment, comment_graph ,label, subgraphs = next(iter(self.dataloaders["train"]))
                    if self.model_type == HYPHEN:
                        self.model.content_encoder._init_hidden_state(len(label))
                        _,_,_ = self.model(
                            content=content, comment=comment_graph, subgraphs=subgraphs
                        )
                    else:
                        _,_,_ = self.model(
                            content=content, comment=comment
                        )
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        prof.export_chrome_trace("trace.json")

    def num_params(self): 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")

        
    def load_model(self):
        dst_dir = f"saved_models/{self.platform}/{self.model_type}"
        os.makedirs(dst_dir, exist_ok=True)
        model_path=f"{dst_dir}/best_model_{self.manifold_name}.pt"
        # self.model = torch.load(model_path)
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")

        
    def build_model( self,
        batch_size=32,
    ):
        self._fit_on_texts()

        print("Building model....")
        if self.model_type == HYPHEN:
            self._build_hyphen(n_classes=2, batch_size=batch_size)
        elif self.model_type == BERT:
            self._build_bert(n_classes=2, batch_size=batch_size)
        elif self.model_type == HAN:
            self._build_han(n_classes=2, batch_size=batch_size)
        else:
            self._build_ssm4rc(n_classes=2, batch_size=batch_size)

        self.model = accelerator.prepare(self.model) 
        self.model.eval()


    @torch.no_grad()
    def _visualize_reconstructions(self, model, dataloader, device, num_imgs: int = 5):
        """ Visualizes image reconstructions of a VAE-model. 
        
        Dataloader has to have a batch_size > num_imgs!

        Returns a matplotlib.pyplot figure.
        """
        model.eval()
        model.to(device)

        x, _ = next(iter(dataloader))

        x = x[:num_imgs] # Select first images
        x = x.to(device)
        x_hat = model.module.reconstruct(x)

        x = x.cpu().detach().numpy()
        x_hat = x_hat.cpu().detach().numpy()

        fig = plt.figure()

        for i in range(num_imgs):
            # Plot input img
            _ = fig.add_subplot(2, num_imgs,i+1, xticks=[], yticks=[])
            plt.imshow(x[i].transpose(1,2,0), cmap='gray')

            # Plot reconstructed img
            _ = fig.add_subplot(2, num_imgs,(i+1)+num_imgs, xticks=[], yticks=[])
            plt.imshow(x_hat[i].transpose(1,2,0), cmap='gray')
            
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        return fig

    @torch.no_grad()
    def _visualize_generations(self, model, device, num_imgs_per_axis: int = 5):
        """ Visualizes image generations of a VAE-model. 

        Returns a matplotlib.pyplot figure.
        """
        model.eval()
        model.to(device)

        x_gen = model.module.generate_random(num_imgs_per_axis**2, device)
        x_gen = x_gen.cpu().detach().numpy()

        fig = plt.figure(figsize=(10,10))
        for i in range(num_imgs_per_axis**2):
            # Plot input img
            ax = fig.add_subplot(num_imgs_per_axis, num_imgs_per_axis, i+1, xticks=[], yticks=[])
            plt.imshow(x_gen[i].transpose(1,2,0), cmap='gray') 

        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
            
        return fig

    @torch.no_grad()
    def visualize_hyperbolic(self, data,  labels=None):
        """ Plots hyperbolic data on Poincaré ball and tangent space 

        Note: This function only supports curvature k=1.
        """

        fig = plt.figure(figsize=(14,7))
        if labels is not None:
            labels = labels.cpu()

        # 2D embeddings
        if (data.shape[-1]==2 and isinstance(self.manifold, PoincareBall) ) or (data.shape[-1]==3 and not isinstance(self.manifold, PoincareBall)):
            if  isinstance(self.manifold, PoincareBall) :
                data_P = data
            else:
                data_P = lorentz_to_poincare(data, k=self.manifold.k).cpu()
        # Dimensionality reduction to 2D
        else:
            if  isinstance(self.manifold, PoincareBall) :
                data = poincare_to_lorentz(data, self.manifold.k)
            reducer = umap.UMAP(output_metric='hyperboloid')
            data = reducer.fit_transform(data.cpu().numpy())
            data = self.manifold.add_time(torch.tensor(data).to(self.device))
            data_P = lorentz_to_poincare(data, k=self.manifold.k).cpu()

        ax = fig.add_subplot(1,2,1)
        plt.scatter(data_P[:,0], data_P[:,1], c=labels, s=1)
        # Draw Poincaré boundary
        boundary=plt.Circle((0,0),1, color='k', fill=False)
        ax.add_patch(boundary)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar()
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        ax.set_title("Poincaré Ball")

        # Plot hyperbolic embeddings in tangent space of the origin
        if  isinstance(self.manifold, PoincareBall) :
            z_all_T = (self.manifold.logmap0(data_P.to(self.device))).detach().cpu()
        else:
            z_all_T = (self.manifold.logmap0(data)).detach().cpu()
            z_all_T = z_all_T[..., 1:]

        ax = fig.add_subplot(1,2,2)
        plt.scatter(z_all_T[:,0], z_all_T[:,1], c=labels, s=1)
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        ax.set_title("Tangent Space")
        return fig

    @torch.no_grad()
    def visualize_embeddings(self):
        """ Visualizes embeddings of a model. 

        Umap only supports k=1?
        """
        self.model.eval()

        s_all = []
        c_all = []
        co_all = []
        labels = []

        for i, sample in enumerate(self.dataloaders["val"]):
            content, comment, _,label, _ = sample
            s ,c, co_sc, a_s, a_c = self.model.forward_features(
                content=content, comment=comment 
            )  # As and Ac are the attention weights we are returning
            labels.append(label.argmax(dim=-1))
            
            co_all.append(co_sc)
            s_all.append(s.view(s.shape[0], -1))
            c_all.append(c.view(c.shape[0], -1))

        s_all = torch.cat(s_all, dim=0) # gpu or cpu
        c_all = torch.cat(c_all, dim=0)# gpu or cpu
        co_all = torch.cat(co_all, dim=0)# gpu or cpu
        labels = torch.cat(labels) # cpu

        if isinstance(self.manifold, PoincareBall) or isinstance(self.manifold, CustomLorentz):
            fig = self.visualize_hyperbolic(co_all, labels)
        else:
            # Plot Euclidean embeddings
            if co_all.shape[-1]>2:
                reducer = umap.UMAP()
                co_all = reducer.fit_transform(co_all)
            else:
                co_all = co_all
            
            fig = plt.figure(figsize=(14, 7))

            ax = fig.add_subplot(1,2,1)
            plt.scatter(co_all[:,0], s_all[:,1], c=labels, s=1)
            ax.set_aspect('equal', adjustable='box')
            plt.colorbar()
            plt.xlabel("$z_0$")
            plt.ylabel("$z_1$")
            
        return fig