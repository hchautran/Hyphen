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
from model.ssm.hs4 import HS4Model 
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
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

    def _build_HS4Model(self, n_classes=2, batch_size=12):
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

        model = HS4Model(
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
            self._build_HS4Model(n_classes=train_y.shape[-1], batch_size=batch_size)

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
        self.train_dataset = FakeNewsDataset(
            content=encoded_train_x,
            comment=encoded_train_c,
            comment_graph=train_c,
            labels=train_y,
            subgraphs=sub_train,
        )
        self.val_dataset = FakeNewsDataset(
            content=encoded_val_x,
            comment=encoded_val_c,
            comment_graph=val_c,
            labels=val_y,
            subgraphs=sub_val,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        self.dataset_sizes = {
            "train": self.train_dataset.__len__(),
            "val": self.val_dataset.__len__(),
        }
        self.train_loader, self.val_loader = accelerator.prepare(self.train_loader, self.val_loader) 
        self.dataloaders = {"train": self.train_loader, "val": self.val_loader}
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
            self._build_HS4Model(n_classes=train_y.shape[-1], batch_size=batch_size)

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
            self._build_HS4Model(n_classes=2, batch_size=batch_size)

        self.model = accelerator.prepare(self.model) 
        self.model.eval()

    
    def fit_reducer(self, data:torch.Tensor, output_metric:str='euclidean', n_neighbors=15):
        reducer = umap.UMAP(output_metric=output_metric, n_neighbors=n_neighbors)
        if  isinstance(self.manifold, PoincareBall) :
            data = self.manifold.expmap0(data)
            self.manifold.assert_check_point_on_manifold(data)
            data = poincare_to_lorentz(data, torch.abs(self.manifold.k)).cpu()

        reducer.fit(data.cpu().numpy())
        return reducer




    @torch.no_grad()
    def visualize_hyperbolic(self, reducer:umap.UMAP, data:torch.Tensor, preds:torch.Tensor=None,labels:torch.Tensor=None, size:int=10, use_label=False):
        """ Plots hyperbolic data on Poincaré ball and tangent space 

        Note: This function only supports curvature k=1.
        """

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1,2,1)
        c_label = labels.cpu() if use_label else preds.cpu()
        lorentz = CustomLorentz(k=1) 
        if  isinstance(self.manifold, PoincareBall) :
            data = self.manifold.expmap0(data)
            self.manifold.assert_check_point_on_manifold(data)
            data = poincare_to_lorentz(data, torch.abs(self.manifold.k)).cpu()
        data = reducer.transform(data.cpu().numpy())
        data = lorentz.add_time(torch.tensor(data).to(self.device))
        data_P = lorentz_to_poincare(data, k=torch.abs(self.manifold.k)).cpu()
        
        plt.scatter(data_P[:,0], data_P[:,1], c=c_label, s=size, cmap='coolwarm', alpha=0.5)
        # Add legend
        plt.colorbar(label="Labels")
        plt.legend()
        # Draw Poincaré boundary
        # Create a grid of points inside a circle
        n = 500 
        x = np.linspace(-1.5, 1.5, n)
        y = np.linspace(-1.5, 1.5, n)
        X, Y = np.meshgrid(x, y)

        Z = np.sqrt(X**2 + Y**2) 

        Z = Z * 2.0/ np.max(Z)

        # Mask values outside the circle
        mask = Z <= 1  # Only show inside the circle

        # Set the background color for outside the circle
        gradient_color = np.ones_like(Z)
        gradient_color[mask] = Z[mask]

        # Plot the gradient using imshow
        ax.imshow(gradient_color, extent=(-1.5, 1.5, -1.5, 1.5), origin='upper', cmap='pink')

        # Add a circular boundary
        circle = plt.Circle((0, 0), 1.0, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim([-1.25, 1.25])
        ax.set_ylim([-1.25, 1.25])
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # plt.xlabel("$z_0$")
        # plt.ylabel("$z_1$")
        # ax.set_title("Poincaré Ball")

        # Plot hyperbolic embeddings in tangent space of the origin
        # if  isinstance(self.manifold, PoincareBall) :
        #     z_all_T = (self.manifold.logmap0(data_P.to(self.device))).detach().cpu()
        # else:
        #     z_all_T = (self.manifold.logmap0(data)).detach().cpu()
        #     z_all_T = z_all_T[..., 1:]

        # ax = fig.add_subplot(1,2,2)
        # plt.scatter(z_all_T[:,0], z_all_T[:,1], c=labels, s=1)
        # ax.set_aspect('equal', adjustable='box')
        # plt.colorbar()
        # plt.xlabel("$z_0$")
        # plt.ylabel("$z_1$")
        # ax.set_title("Tangent Space")
        return fig

    def visualize_euclid(self, reducer:umap.UMAP, data:torch.Tensor, preds:torch.Tensor=None,labels:torch.Tensor=None, size:int=10, use_label=False):
        data = data.cpu()
        data = reducer.transform(data)
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1,2,1)
        plt.scatter(data[:,0], data[:,1], c=labels.cpu() if use_label else preds.cpu(), s=size, cmap='coolwarm')
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")

    @torch.no_grad()
    def visualize_embeddings(self):
        """ Visualizes embeddings of a model. 

        Umap only supports k=1?
        """
        self.model.eval()

        sent_s_all = []
        sent_c_all = []
        word_s_all = []
        word_c_all = []
        co_s_all = []
        co_c_all = []
        co_all = []
        labels = []
        sent_labels= []
        word_labels= []
        pred_labels= []
        pred_sent_labels= []
        pred_word_labels= []

        for i, sample in enumerate(self.dataloaders["val"]):
            content, comment, _,label, _ = sample
            word_s, word_c, sent_s , sent_c, co_s, co_c, co_sc, preds = self.model.forward_features(
                content=content, comment=comment 
            )  # As and Ac are the attention weights we are returning

            label = label.argmax(dim=-1)
            batch_idx = torch.arange(label.shape[0])
            preds = preds[batch_idx, 1] 
            labels.append(label)
            pred_labels.append(preds)
            # print(label.shape)
            # print(label[..., None].expand(s.shape[0],s.shape[1]).shape)
            sent_labels.append(label[..., None].expand(sent_s.shape[0], sent_s.shape[1]))
            word_labels.append(label[..., None, None].expand(word_s.shape[0], word_s.shape[1], word_s.shape[2]))
            pred_sent_labels.append(preds[..., None].expand(sent_s.shape[0], sent_s.shape[1]))
            pred_word_labels.append(preds[..., None, None].expand(word_s.shape[0], word_s.shape[1], word_s.shape[2]))
            sent_s_all.append(sent_s.view(-1,sent_s.shape[-1]))
            sent_c_all.append(sent_c.view(-1,sent_c.shape[-1]))
            word_s_all.append(word_s.view(-1,word_s.shape[-1]))
            word_c_all.append(word_c.view(-1,word_c.shape[-1]))
            co_s_all.append(co_s.view(-1,sent_s.shape[-1]))
            co_c_all.append(co_c.view(-1,sent_c.shape[-1]))
            co_all.append(co_sc)

        word_s_all = torch.cat(word_s_all, dim=0) # gpu or cpu
        word_c_all = torch.cat(word_c_all, dim=0)# gpu or cpu
        sent_s_all = torch.cat(sent_s_all, dim=0) # gpu or cpu
        sent_c_all = torch.cat(sent_c_all, dim=0)# gpu or cpu
        co_s_all = torch.cat(co_s_all, dim=0) # gpu or cpu
        co_c_all = torch.cat(co_c_all, dim=0)# gpu or cpu
        co_all = torch.cat(co_all, dim=0)# gpu or cpu

        labels = torch.cat(labels) # cpu
        pred_labels = torch.cat(pred_labels) # cpu
        sent_labels = torch.cat(sent_labels) # cpu
        word_labels = torch.cat(word_labels) # cpu
        pred_sent_labels = torch.cat(pred_sent_labels) # cpu
        pred_word_labels = torch.cat(pred_word_labels) # cpu

        # print('sent_labels', sent_labels.shape)
        # print('word_labels', word_labels.shape)
        # print('sent_s_all', sent_s_all.shape)
        # print('sent_c_all', sent_c_all.shape)
        # print('word_s_all', word_s_all.shape)
        # print('word_c_all', word_c_all.shape)
        # print('co_s_all', co_s_all.shape)
        # print('co_c_all', co_c_all.shape)
        # print('co_all', co_all.shape)
        # print('labels', labels.shape)   
        
        print('fitting document-level reducer')
        output_metric = 'hyperboloid' if self.manifold_name != EUCLID else 'euclidean'
        doc_level_reducer = self.fit_reducer(co_all, output_metric=output_metric)

        print('fitting sent-level reducer')
        # co_sent_level_reducer = self.fit_reducer(torch.cat([co_c_all, co_s_all], dim=0), output_metric=output_metric)
        sent_level_reducer = self.fit_reducer(torch.cat([sent_s_all, sent_c_all], dim=0), output_metric=output_metric)  

        print('fitting word-level reducer')
        # word_embeds = torch.cat([word_s_all, word_c_all])
        if isinstance(self.manifold, CustomLorentz):
            # clip_r = 2.0 
            # if isinstance(self.manifold, PoincareBall):
            #     x_norm = torch.norm(word_embeds, dim=-1, keepdim=True) + 1e-5
            #     fac = torch.minimum(torch.ones_like(x_norm), clip_r/ x_norm)
            #     word_embeds = word_embeds * fac
            #     word_embeds = self.manifold.expmap0(word_embeds)
            # elif isinstance(self.manifold, CustomLorentz):
            #     x_norm = torch.norm(word_embeds, dim=-1, keepdim=True) + 1e-5
            #     fac = torch.minimum(torch.ones_like(x_norm), clip_r/ x_norm)
            #     word_embeds = word_embeds * fac
            #     word_embeds = F.pad(word_embeds, (1,0), "constant", 0)
            #     word_embeds = self.manifold.expmap0(word_embeds)
            # word_level_reducer = self.fit_reducer(word_embeds, output_metric='hyperboloid')
            fig_co = self.visualize_hyperbolic(doc_level_reducer, data=co_all, preds=pred_labels,labels=labels, size=50, use_label=True)
            fig_co_truth = self.visualize_hyperbolic(doc_level_reducer, data=co_all, preds=pred_labels,labels=labels, size=50, use_label=False)
            # fig_co_s = self.visualize_hyperbolic(co_sent_level_reducer, data=torch.cat([co_s_all]), preds=pred_labels,labels=torch.cat([labels]), size=50)
            # fig_co_c = self.visualize_hyperbolic(co_sent_level_reducer, data=torch.cat([co_c_all]), preds=pred_labels,labels=torch.cat([labels]), size=50)
            fig_sent_s = self.visualize_hyperbolic(sent_level_reducer, data=torch.cat([sent_s_all]), preds=pred_sent_labels,labels=torch.cat([sent_labels]), size=10)
            fig_sent_c = self.visualize_hyperbolic(sent_level_reducer, data=torch.cat([sent_c_all]),  preds=pred_sent_labels,labels=torch.cat([sent_labels]), size=10)
        elif isinstance(self.manifold, PoincareBall):
            fig_co = self.visualize_euclid(doc_level_reducer, data=co_all, labels=labels, preds=pred_labels,size=50, use_label=True)
            fig_co_truth = self.visualize_euclid(doc_level_reducer, data=co_all, preds=pred_labels,labels=labels, size=50, use_label=False)
            # fig_co_s = self.visualize_euclid(co_sent_level_reducer, data=torch.cat([co_s_all]), preds=pred_labels,labels=torch.cat([labels]), size=50)
            # fig_co_c = self.visualize_euclid(co_sent_level_reducer, data=torch.cat([co_c_all]), preds=pred_labels,labels=torch.cat([labels]), size=50)
            fig_sent_s = self.visualize_euclid(sent_level_reducer, data=torch.cat([sent_s_all]), preds=pred_sent_labels, labels=torch.cat([sent_labels]), size=10)
            fig_sent_c = self.visualize_euclid(sent_level_reducer, data=torch.cat([sent_c_all]),  preds=pred_sent_labels, labels=torch.cat([sent_labels]), size=10)
        else:
            # Plot Euclidean embeddings
            print('fitting word-level reducer')
            fig_co = self.visualize_euclid(doc_level_reducer, co_all, labels, size=50)
            fig_co_truth = self.visualize_euclid(doc_level_reducer, data=co_all, preds=pred_labels,labels=labels, size=50, use_label=False)
            # fig_co_s = self.visualize_euclid(co_sent_level_reducer, torch.cat([co_s_all]), torch.cat([labels]), size=50)
            # fig_co_c = self.visualize_euclid(co_sent_level_reducer, torch.cat([co_c_all]), torch.cat([labels]), size=50)
            fig_sent_s = self.visualize_euclid(sent_level_reducer, torch.cat([sent_s_all]),  torch.cat([sent_labels]), size=10)
            fig_sent_c = self.visualize_euclid(sent_level_reducer, torch.cat([sent_c_all]),  torch.cat([sent_labels]), size=10)
                
        # return fig_co, fig_co_s, fig_co_c, fig_sent_s, fig_sent_c 
        return fig_co, fig_co_truth
        # return fig_word