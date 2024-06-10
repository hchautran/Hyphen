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
from hyptorch.geoopt.optim import RiemannianAdam
from utils.metrics import Metrics
from model.model import  Model
from utils.dataset import FakeNewsDataset
from utils.utils import get_evaluation
import wandb
from transformers import AutoTokenizer
from accelerate import Accelerator
from model import (
    EuclidS4Enc,
    EuclidAMRComEnc,
    EuclidCoAttention,
    EuclidGRUPostEnc,
    HybridAMRComEnc,
    HybridGRUPostEnc,
    HybridCoAttention,
    LorentzCoAttention
    
)
from const import *
import os

accelerator = Accelerator()
DATA_PATH = os.getcwd() 

class HyphenTrainer:
    def __init__(
        self,
        lr,
        n_classes:int,
        manifold:str,
        content_module:nn.Module,
        comment_module:nn.Module,
        fourier:bool,
        train_dataset:FakeNewsDataset,
        val_dataset:FakeNewsDataset,
        word_hidden_size, 
        sent_hidden_size, 
        max_sent_length, 
        max_word_length, 
        graph_hidden,
        log_enable:bool=False,
        batch_size:int=32,
        curv=1.0,
        max_sentence_count = 50, 
        max_comment_count = 10, 
        embedding_dim = 100, 
        latent_dim = 100, 
        graph_glove_dim = 100,
    ):
        self.curv=curv
        self.n_classes=n_classes
        self.max_sents = max_sentence_count
        self.max_coms = max_comment_count
        self.embedding_dim=embedding_dim
        self.latent_dim= latent_dim 
        self.graph_glove_dim = graph_glove_dim 
        self.latent_dim = latent_dim
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.graph_hidden = graph_hidden
        self.batch_size = batch_size

        self.vocab_size = 0
        self.metrics = Metrics()
        self.device = accelerator.device
        self.manifold = manifold
        self.lr = lr
        self.content_module = content_module
        self.comment_module = comment_module
        self.fourier = fourier
        self.platform = train_dataset.name 
        self.log_enable = log_enable 
        self.train_dataset = train_dataset
        if self.log_enable:
            wandb.init(
                project='SSM4CTC',
                name=f'{self.platform}_{manifold}',
                config={
                    'dataset': self.platform,
                    'manifold': manifold,
                }
            )
            wandb.watch(self.model, log='all')
        
        self.tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataset = train_dataset,
        self.val_dataset = val_dataset,
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )

        print("Building model....")
        self.model = self._build_model()
        model = accelerator.prepare(model) 
        self.optimizer = RiemannianAdam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader, self.val_loader = accelerator.prepare(
            self.train_loader, self.val_loader, self.optimizer
        )
            

    def log(self, stats):
        if self.log_enable:
            wandb.log(stats)


    def _build_comment_module(self, manifold:str)->nn.Module:
        if manifold ==  LORENTZ:
            pass
        elif manifold == POINCARE: 
            return HybridAMRComEnc(
                in_dim=self.graph_glove_dim,
                hidden_dim=self.graph_hidden,
                max_comment_count=self.max_coms,
                curv=self.curv
            ) 
        else:
            return HybridAMRComEnc(
                in_dim=self.graph_glove_dim,
                hidden_dim=self.graph_hidden,
                max_comment_count=self.max_coms,
                curv=self.curv
            ) 

    def _build_post_module(self)->nn.Module:
        if self.manifold == LORENTZ:
            pass 
        elif self.manifold == POINCARE: 
            return HybridGRUPostEnc(
                word_hidden_size=self.word_hidden_size,
                sent_hidden_size=self.sent_hidden_size,
                batch_size=self.batch_size

            ) 
        else:
            return  

    def _build_coattention_module(self)->nn.Module:
        if self.manifold == LORENTZ:
            return LorentzCoAttention(
                latent_dim=self.latent_dim,
                embedding_dim=self.embedding_dim,
                embedding_dim=self.embedding_dim,
            )
        elif self.manifold == POINCARE: 
            return 
        else:
            return 

    def _build_fc(self)->nn.Module:
        if self.manifold == LORENTZ:
            pass
        elif self.manifold == POINCARE: 
            pass
        else:
            pass

    def _build_model(self) :
        """
        This function is used to build Hyphen model.
        """
        embeddings_index = {}
        self.glove_dir = f"{DATA_PATH}/glove.twitter.27B.100d.txt"
        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            

        f.close()
        word_index = self.tokenizer.vocab
        embedding_matrix = np.random.random((len(word_index) + 1, self.embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


        model = Model(
            embedding_matrix=embedding_matrix,
            word_hidden_size=self.word_hidden_size,
            sent_hidden_size=self.sent_hidden_size,
            max_sent_length=self.max_sent_length,
            max_word_length=self.max_word_length,
            graph_hidden=self.graph_hidden,
            batch_size=self.batch_size,
            num_classes=self.n_classes,
            max_comment_count=self.max_coms,
            max_sentence_count=self.max_sents,
            manifold=self.manifold,
            comment_module=self.comment_module,
            content_module=self.content_module,
        )
        print("Model built")


        return model



    def test(
        self,
    ):

        self.dataset_sizes = {
            "train": self.train_dataset.__len__(),
            "val": self.val_dataset.__len__(),
        }

        self.model.load_state_dict(
            torch.load(f"saved_models/{self.platform}/best_model_{self.manifold}.pt")
        )

        print("Loaded state dict")

        self.model.eval()
        loss_ls = []
        total_samples = 0
        As_batch, Ac_batch, predictions_batch = [], [], []
        for i, sample in enumerate(self.val_loader):
            content, comment, label, subgraphs = sample
            num_sample = len(label)  # last batch size
            total_samples += num_sample


            self.model.content_encoder._init_hidden_state(num_sample)

            predictions, As, Ac = self.model(content, comment, subgraphs)

            te_loss = self.criterion(predictions, label)
            loss_ls.append(te_loss * num_sample)

            _, predictions = torch.max(torch.softmax(predictions, dim=-1), 1)
            _, label = torch.max(label, 1)

            As_batch.extend(As.detach().cpu().numpy())
            Ac_batch.extend(Ac.detach().cpu().numpy())
            predictions_batch.extend(predictions.detach().cpu().numpy())
        return predictions_batch, As_batch, Ac_batch

    def train(
        self,
        epochs=5,
    ):

   

        self.run_epoch(epochs)

        
    def run_epoch(self, epochs):
        """
        Function to train model for given epochs
        """
        best_f1 = 0.0


        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 100)
            self.metrics.on_train_begin()
            self.model.train()

            num_iter_per_epoch = len(self.train_loader)
            for iter, sample in enumerate(tqdm.tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                content, _, comment, label, subgraphs = sample

                self.model.content_encoder._init_hidden_state(len(label))
                predictions = self.model(
                    content, comment, subgraphs
                )  # As and Ac are the attention weights we are returning
                loss = self.criterion(predictions, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                training_metrics = get_evaluation(
                    torch.max(label, 1)[1].cpu().numpy(),
                    predictions.cpu().detach().numpy(),
                    list_metrics=["accuracy"],
                )
                self.log({
                    "Train/Loss": loss, 
                    "Train/Epoch": epoch * num_iter_per_epoch + iter,
                    "Train/Accuracy":training_metrics["accuracy"],
                })

            self.model.eval()
            loss_ls = []
            total_samples = 0
            for i, sample in enumerate(self.val_loader):
                content, _ ,comment, label, subgraphs = sample
                num_sample = len(label)  # last batch size
                total_samples += num_sample
                self.model.content_encoder._init_hidden_state(num_sample)
                predictions = self.model(content, comment, subgraphs)

                te_loss = self.criterion(predictions, label)
                loss_ls.append(te_loss * num_sample)

                _, predictions = torch.max(torch.softmax(predictions, dim=-1), 1)
                _, label = torch.max(label, 1)

                predictions = predictions.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                self.metrics.on_batch_end(epoch, i, predictions, label)

            acc, f1, precision, recall = self.metrics.on_epoch_end(epoch)
            if f1 > best_f1:
                print(f"Best F1: {f1}")
                print("Saving best model!")
                self.log({'epoch':epoch, 'best F1': f1, 'best precision': predictions, 'best recall': recall})
                dst_dir = f"saved_models/{self.platform}/"
                os.makedirs(dst_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(), f"{dst_dir}best_model_{self.manifold}.pt"
                )
                best_f1 = f1

            te_loss = sum(loss_ls) / total_samples
            self.log({
                "Test/epoch": epoch,
                "Test/Loss": te_loss,
                "Test/Accuracy": acc,
                "Test/F1": f1,
                "Test/Precision": precision,
                "Test/Recall": recall,
                
            })

        print(f"Best F1: {best_f1}")
        print("Training  end")
        print("-" * 100)

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
