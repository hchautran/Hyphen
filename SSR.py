
import dgl
import os
import csv
import sys
import time
import tqdm
import wandb
import torch
import pickle
import sklearn
import warnings
import numpy as np
import torch.nn as nn
from utils import manifolds
from lcoattention import CoAttention 
from utils.layers.hyp_layers import *
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LMLR import LorentzMLR 
from transformers import DistilBertConfig, DistilBertModel
from utils.layers.attn_layers import GraphAttentionLayer 
from torch.utils.data import DataLoader
from hyptorch.geoopt.optim import RiemannianAdam
from utils.metrics import Metrics
from utils.dataset import FakeNewsDataset
from utils.utils import get_evaluation
from transformers import AutoTokenizer
from accelerate import Accelerator
from utils.manifolds import PoincareBall
from s4d import S4D
from torch.utils.data.dataset import Dataset

class FakeNewsDataset(Dataset):

    def __init__(self, content, comment, labels,   max_length_sentences=30, max_length_word=35):
        super(FakeNewsDataset, self).__init__()

        self.content = content
        self.comment = comment
        self.labels = labels

        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word

        self.num_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, samples):
        """Here, samples will be the sample returned by __getitem__ function"""
        content, comment, label, subgraphs = map(list, zip(*samples))
        content = torch.from_numpy(np.array(content))
        label = torch.from_numpy(np.array(label))
        return content, comment, label

    def __getitem__(self, index):
        return self.content[index], self.comment[index], self.labels[index], self.subgraphs[index]


csv.field_size_limit(sys.maxsize)

if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

DATA_PATH = os.getcwd()
accelerator = Accelerator()

class S4Model(nn.Module):

    def __init__(
        self,
        d_input=100,
        d_model=256,
        d_output=100,
        n_layers=1,
        dropout=0.1,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.0001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Linear(d_model, d_output)


    def forward(self, x:torch.Tensor, pooling=False):
        """
        Input x is shape (B, L, d_input)
        """
        # print(x.shape)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        

        # Pooling: average pooling over the sequence length
        if pooling:
            x = x.mean(dim=1)
        x = self.decoder(x)  #

        return x


class SSRComEnc(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        max_comment_count,
        device,
        manifold,
        content_module,
        comment_curvature,
    ):
        super(SSRComEnc, self).__init__()
        self.manifold = manifold
        self.c = comment_curvature
        self.max_comment_count = max_comment_count
        self.hidden_dim = hidden_dim
        self.device = device
        self.content_module = content_module
    
    
    def forward(self, g, h, subgraphs):
        """returned shape will be [batch_size, max_comments, embedding_size] i.e.
        [batch_size, max_comment_count, hidden_dim]"""

        # Apply graph convolution and activation.
        adj = g.adj().to(self.device)  # finding the adjacency matrix
        inp = h.to(self.device)  # convertng to sparse tensor
        print(h.shape)

        if isinstance(self.manifold, PoincareBall):
            inp = torch.cat(
                [
                    self.manifold.proj(
                        self.manifold.expmap0(
                            self.manifold.proj_tan0(i, c=self.c), c=self.c
                        ),
                        c=self.c,
                    ).unsqueeze(0)
                    for i in inp
                ],
                axis=0,
            )

        out, adj = self.conv1((inp, adj))
        out, adj = self.conv2((out, adj))
        h = out  # converting back to dense
        h = self.manifold.logmap0(self.manifold.proj(h, c=self.c), c=self.c)
        # map h (which is in poincare space/euclidean) to tangential space to aggregate the node representations
        if self.content_module:
            with g.local_scope():
                g.ndata["h"] = h
                # Calculate graph representation by average readout.
                unbatched = dgl.unbatch(g)
                batch_agg = []
                for batch_idx in range(len(unbatched)):
                    agg = []
                    for node_list in subgraphs[batch_idx]:
                        sub = dgl.node_subgraph(unbatched[batch_idx], node_list)
                        hg = dgl.mean_nodes(sub, "h")
                        agg.append(torch.squeeze(hg).unsqueeze(0))
                    if len(agg) >= self.max_comment_count:
                        agg = agg[: self.max_comment_count]
                        agg = torch.cat([i.float() for i in agg], dim=0)
                    else:
                        padding = torch.zeros(
                            (self.max_comment_count - len(agg), self.hidden_dim),
                            dtype=torch.float32,
                            requires_grad=True,
                        ).to(self.device)
                        without_padding = torch.cat([i.float() for i in agg], dim=0)
                        agg = torch.cat([without_padding, padding], dim=0)
                    agg = self.manifold.proj(
                        self.manifold.expmap0(agg, c=self.c), c=self.c
                    )
                    batch_agg.append(agg.unsqueeze(0))
                ret = torch.cat(batch_agg, dim=0)
                return ret

        else:
            with g.local_scope():
                g.ndata["h"] = h
                ret = dgl.mean_nodes(g, "h")
                ret = self.manifold.proj(self.manifold.expmap0(ret, c=self.c), c=self.c)
                return ret



# eps = 1e-7
# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) 

class SSRPost(nn.Module):
    def __init__(
        self, 
        word_hidden_size, 
        sent_hidden_size, 
        batch_size, 
        num_classes, 
        embedding_matrix, 
        max_sent_length, 
        max_word_length, 
        device, 
        manifold,
        content_curvature
    ):
        super(SSRPost, self).__init__()
        self.batch_size = batch_size
        self.device = device
       
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.manifold = manifold
        self.content_curvature = content_curvature

        self.word_ssm= S4Model(d_input=word_hidden_size, d_model=word_hidden_size*4, d_output=sent_hidden_size)
        self.sent_ssm= S4Model(d_input=sent_hidden_size, d_model=sent_hidden_size*4, d_output=sent_hidden_size)
        self.lookup = self.create_embeddeding_layer(embedding_matrix)

    def forward(self, input):
        output_list = []
        # input = input.permute(1, 0, 2)
        for x in input:
            x = self.lookup(x)
            output = self.word_ssm(x=x, pooling=True) 
            output_list.append(output)
        output = torch.stack(output_list, dim=0)
        output = self.sent_ssm(output)
        output = self.manifold.expmap0(output, c=1.0)
        return output

    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer


        


class SSR(nn.Module):

    def __init__(
        self, 
        embedding_matrix,  
        word_hidden_size, 
        sent_hidden_size, 
        max_sent_length, 
        max_word_length, 
        device, 
        graph_hidden, 
        num_classes = 2, 
        max_sentence_count = 50 , 
        max_comment_count = 10, 
        batch_size = 32 ,
        embedding_dim = 100, 
        latent_dim = 100, 
        graph_glove_dim = 100, 
        manifold = "hyper", 
        use_gat=False,
        content_module =True, 
        comment_module = True, 
        fourier = False
    ):
        super(SSR,self).__init__()

        self.device = device
        self.fourier = fourier
        self.comment_curvature = torch.tensor(1.0)
        self.content_curvature = torch.tensor(1.0)
        self.combined_curvature = torch.tensor(1.0)
        self.graph_glove_dim = graph_glove_dim#the dimension of glove embeddings used to initialise the comments amr graph
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_sentence_count = max_sentence_count
        self.max_comment_count = max_comment_count
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length  = max_word_length 
        self.graph_hidden = graph_hidden
        self.manifold = getattr(manifolds, 'PoincareBall')()
        self.lorentz = CustomLorentz(k=1)
        self.comment_module = comment_module
        self.content_module = content_module 

        self.content_encoder= SSRPost(
            word_hidden_size=self.word_hidden_size, 
            sent_hidden_size=self.sent_hidden_size, 
            batch_size=batch_size, 
            num_classes=num_classes, 
            embedding_matrix=embedding_matrix, 
            max_sent_length=self.max_sent_length, 
            max_word_length=self.max_word_length, 
            device=self.device, 
            manifold=self.manifold, 
            content_curvature=self.content_curvature
        )
        self.comment_encoder = SSRPost(
            word_hidden_size=self.word_hidden_size, 
            sent_hidden_size=self.sent_hidden_size, 
            batch_size=batch_size, 
            num_classes=num_classes, 
            embedding_matrix=embedding_matrix, 
            max_sent_length=self.max_sent_length, 
            max_word_length=self.max_word_length, 
            device=self.device, 
            manifold=self.manifold, 
            content_curvature=self.content_curvature
        )
        self.coattention = CoAttention(manifold=self.manifold, embedding_dim=self.embedding_dim, fourier=self.fourier, combined_curvature=self.combined_curvature)
        
        if self.comment_module and self.content_module: 
            self.fc = nn.Sequential( 
                # LFC_Block(self.manifold, 2*latent_dim + 1, 2*latent_dim + 1, normalization="batch_norm"),
                LorentzMLR(self.lorentz, 2*latent_dim + 1, num_classes),
            )
        elif self.comment_module: 
            self.fc = nn.Sequential( 
                # LFC_Block(self.manifold, latent_dim + 1, latent_dim + 1, normalization="batch_norm"),
                LorentzMLR(self.lorentz, latent_dim + 1, num_classes),
            )
        else: 
            self.fc = nn.Sequential( 
                LorentzMLR(self.lorentz, latent_dim + 1, num_classes),
            )

    def forward(self, content, comment, subgraphs):

        content_embedding = self.content_encoder(content)
        comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
        assert not torch.isnan(content_embedding).any(), "content_embedding is nan"
        assert not torch.isnan(comment_embedding).any(), "comment_embedding is nan"
        coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
        preds = self.fc(coatten)
        return preds


class SSRModel:
    def __init__(
        self,
        platform,
        max_sen_len,
        max_com_len,
        max_sents,
        max_coms,
        manifold,
        lr,
        content_module,
        comment_module,
        fourier,
        use_gat=False,
        log_enable=False,
    ):
        self.device = accelerator.device
        self.model = None
        self.use_gat = use_gat
        self.max_sen_len = max_sen_len
        self.max_sents = max_sents
        self.max_coms = max_coms
        self.max_com_len = max_com_len
        self.vocab_size = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.sentence_comment_co_model = None
        self.tokenizer = None
        self.metrics = Metrics()
        # self.device = (
            # torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # )
        self.manifold = manifold
        self.lr = lr
        self.content_module = content_module
        self.comment_module = comment_module
        self.fourier = fourier
        self.platform = platform
        self.log_enable = log_enable 
        if self.log_enable:
            wandb.init(
                project='Hyphat',
                name=f'{platform}_{manifold}_{"gcn" if not use_gat else "gat"}',
                config={
                    'dataset': platform,
                    'type': manifold,
                    'use gat': use_gat,  
                    'model':'SSR'
                }
            )
            

    def log(self, stats):
        if self.log_enable:
            wandb.log(stats)
            

    def _fit_on_texts(self, train_x, val_x):
        """
        Creates vocabulary set from the news content and the comments
        """
        texts = []
        texts.extend(train_x)
        texts.extend(val_x)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        print("saved tokenizer")



    def _build_model(self, n_classes=2, batch_size=12, embedding_dim=100):
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

        # get word index
        word_index = self.tokenizer.vocab
        embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        (
            self.word_hidden_size,
            self.sent_hidden_size,
            self.max_sent_length,
            self.max_word_length,
            self.graph_hidden,
        ) = (100, 100, 50, 50, 100)

        model = SSR(
            device=self.device,
            manifold=self.manifold,
            embedding_matrix=embedding_matrix,
            word_hidden_size=self.word_hidden_size,
            sent_hidden_size=self.sent_hidden_size,
            max_sent_length=self.max_sent_length,
            max_word_length=self.max_word_length,
            graph_hidden=self.graph_hidden,
            batch_size=batch_size,
            num_classes=n_classes,
            max_comment_count=self.max_coms,
            max_sentence_count=self.max_sents,
            comment_module=self.comment_module,
            content_module=self.content_module,
            fourier=self.fourier,
            use_gat=self.use_gat
        )
        print("SSR built")

        model = accelerator.prepare(model) 
        self.optimizer = RiemannianAdam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', patience=5, min_lr=1e-7)
        self.optimizer, self.scheduler  = accelerator.prepare(self.optimizer, self.scheduler)
        self.criterion = nn.CrossEntropyLoss()
        return model

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


    def test(
        self,
        train_x,
        train_y,
        train_c,
        val_c,
        val_x,
        val_y,
        sub_train,
        sub_val,
        batch_size=9,
    ):
        self.tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
        print("Building model....")
        self.model = self._build_model(
            n_classes=train_y.shape[-1], batch_size=batch_size, embedding_dim=100
        )
        if self.log_enable:
            wandb.watch(self.model, log='all')

        print("Model built.")

        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x)
        encoded_val_x = self._encode_texts(val_x)
        print("preparing dataset....")

        # adding self loops in the dgl graphs
        train_c = [dgl.add_self_loop(i) for i in train_c]
        val_c = [dgl.add_self_loop(i) for i in val_c]

        train_dataset = FakeNewsDataset(
            encoded_train_x,
            train_c,
            train_y,
            sub_train,
            self.glove_dir,
            self.max_sent_length,
            self.max_word_length,
        )
        val_dataset = FakeNewsDataset(
            encoded_val_x,
            val_c,
            val_y,
            sub_val,
            self.glove_dir,
            self.max_sent_length,
            self.max_word_length,
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

        val_loader, train_loader = accelerator.prepare(val_loader, train_loader)
        self.dataset_sizes = {
            "train": train_dataset.__len__(),
            "val": val_dataset.__len__(),
        }
        self.dataloaders = {"train": train_loader, "val": val_loader}
        print("Dataset prepared.")

        self.model.load_state_dict(
            torch.load(f"saved_models/{self.platform}/best_model_{self.manifold}.pt")
        )

        print("Loaded state dict")

        self.model.eval()
        loss_ls = []
        total_samples = 0
        As_batch, Ac_batch, predictions_batch = [], [], []
        for i, sample in enumerate(self.dataloaders["val"]):
            content, comment, label, subgraphs = sample
            num_sample = len(label)  # last batch size
            total_samples += num_sample
            # comment = comment.to(self.device)
            # content = content.to(self.device)
            # label = label.to(self.device)
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
        train_x,
        train_y,
        train_c,
        val_c,
        val_x,
        val_y,
        sub_train,
        sub_val,
        batch_size=9,
        epochs=5,
    ):

        # Fit the vocabulary set on the content and comments
        self._fit_on_texts(train_x, val_x)

        print("Building model....")
        self.model = self._build_model(
            n_classes=train_y.shape[-1], batch_size=batch_size, embedding_dim=100
        )
        print("Model built.")

        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x)
        # print(encoded_train_x.shape)
        encoded_val_x = self._encode_texts(val_x)
        print("preparing dataset....")

        # adding self loops in the dgl graphs
        train_c = [dgl.add_self_loop(i) for i in train_c]
        val_c = [dgl.add_self_loop(i) for i in val_c]

        train_dataset = FakeNewsDataset(
            encoded_train_x,
            train_c,
            train_y,
            sub_train,
            self.glove_dir,
            self.max_sent_length,
            self.max_word_length,
        )
        val_dataset = FakeNewsDataset(
            encoded_val_x,
            val_c,
            val_y,
            sub_val,
            self.glove_dir,
            self.max_sent_length,
            self.max_word_length,
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

        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

        self.dataset_sizes = {
            "train": train_dataset.__len__(),
            "val": val_dataset.__len__(),
        }
        self.dataloaders = {"train": train_loader, "val": val_loader}
        print("Dataset prepared.")

        # train model for given epoch
        self.run_epoch(epochs)

        
    def run_epoch(self, epochs):
        """
        Function to train model for given epochs
        """

        since = time.time()
        clip = 5  # modify clip

        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_acc = 0.0

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 100)
            self.metrics.on_train_begin()
            self.model.train()

            num_iter_per_epoch = len(self.dataloaders["train"])
            for iter, sample in enumerate(tqdm.tqdm(self.dataloaders["train"])):
                self.optimizer.zero_grad()

                content, comment, label, subgraphs = sample

                # comment = comment.to(self.device)
                # content = content.to(self.device)
                # label = label.to(self.device)
                # self.model.content_encoder._init_hidden_state(len(label))
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
            for i, sample in enumerate(self.dataloaders["val"]):
                content, comment, label, subgraphs = sample
                num_sample = len(label)  # last batch size
                total_samples += num_sample

                # comment = comment.to(self.device)
                # content = content.to(self.device)
                # label = label.to(self.device)

                # self.model.content_encoder._init_hidden_state(num_sample)

                predictions = self.model(content, comment, subgraphs)

                te_loss = self.criterion(predictions, label)
                loss_ls.append(te_loss * num_sample)

                _, predictions = torch.max(torch.softmax(predictions, dim=-1), 1)
                _, label = torch.max(label, 1)

                predictions = predictions.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                self.metrics.on_batch_end(epoch, i, predictions, label)

            acc_, f1 = self.metrics.on_epoch_end(epoch)
            self.scheduler.step(f1)
            if f1 > best_f1:
                print(f"Best F1: {f1}")
                print("Saving best model!")
                self.log({'best F1': f1})
                dst_dir = f"saved_models/{self.platform}/"
                os.makedirs(dst_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(), f"{dst_dir}best_model_{self.manifold}.pt"
                )
                best_model = self.model
                best_f1 = f1

            te_loss = sum(loss_ls) / total_samples
            self.log({
                "Test/epoch": epoch,
                "Test/Loss": te_loss,
                "Test/Accuracy": acc_,
                "Test/F1": f1,
                
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
