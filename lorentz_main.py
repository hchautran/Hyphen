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
from lorentz_hyphen import Hyphen
from utils.dataset import FakeNewsDataset
from utils.utils import get_evaluation
import wandb
from transformers import AutoTokenizer
from const import DATA_PATH
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


class HyphenModel:
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
    ):
        self.model = None
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
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.manifold = manifold
        self.lr = lr
        self.content_module = content_module
        self.comment_module = comment_module
        self.fourier = fourier
        self.platform = platform
     
        self.log_enable = False 
        if self.log_enable:
            wandb.init(
                project='Hyphen',
                name=f'{platform}_{manifold}',
                config={
                    'dataset': platform,
                    'type': manifold
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
        tokenizer = Tokenizer(models.WordLevel())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.WordLevelTrainer(
            vocab_size=500000,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<PAD>", "<BOS>", "<EOS>"],
        )
        all_sentences = []
        for text in texts:
            for sentence in text:
                all_sentences.append(sentence)
        print(len(all_sentences))

        tokenizer.train_from_iterator(all_sentences, trainer=trainer)

        self.tokenizer = tokenizer 

        print("saved tokenizer")

   


    def _build_model(self, n_classes=2, batch_size=12, embedding_dim=100):
        """
        This function is used to build Hyphen model.
        """
        embeddings_index = {}

        self.glove_dir = f"{DATA_PATH}/glove.twitter.27B.100d.txt"
        # self.glove_dir = f"{DATA_PATH}/poincare_glove_100D_cosh-dist-sq_init_trick.txt"

        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            

        f.close()

        # get word index
        word_index = self.tokenizer.get_vocab()
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
        ) = (50, 50, 50, 50, 100)
        model = Hyphen(
            embedding_matrix,
            self.word_hidden_size,
            self.sent_hidden_size,
            self.max_sent_length,
            self.max_word_length,
            self.device,
            graph_hidden=self.graph_hidden,
            batch_size=batch_size,
            num_classes=n_classes,
            max_comment_count=self.max_coms,
            max_sentence_count=self.max_sents,
            manifold=self.manifold,
            comment_module=self.comment_module,
            content_module=self.content_module,
            fourier=self.fourier,
        )
        print("Hyphen built")

        model = model.to(self.device)

        self.optimizer = RiemannianAdam(model.parameters(), lr=self.lr)

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
            tokens = self.tokenizer.encode_batch(text)
            # ids = np.array(tokens[0].ids)

            encoded_text = np.concatenate([np.array(
                torch.nn.functional.pad(
                    torch.tensor(tokens[i].ids),
                    pad=(0, self.max_sen_len - torch.tensor(tokens[i].ids).shape[0]), 
                    mode='constant', 
                    value=0
                )
            )[:self.max_sents][np.newaxis,...] for i in range(len(tokens))])
            print(encoded_text.shape)
            
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

            comment = comment.to(self.device)
            content = content.to(self.device)
            label = label.to(self.device)

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

                comment = comment.to(self.device)
                content = content.to(self.device)
                label = label.to(self.device)
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
            for i, sample in enumerate(self.dataloaders["val"]):
                content, comment, label, subgraphs = sample
                num_sample = len(label)  # last batch size
                total_samples += num_sample

                comment = comment.to(self.device)
                content = content.to(self.device)
                label = label.to(self.device)

                self.model.content_encoder._init_hidden_state(num_sample)

                predictions = self.model(content, comment, subgraphs)

                te_loss = self.criterion(predictions, label)
                loss_ls.append(te_loss * num_sample)

                _, predictions = torch.max(torch.softmax(predictions, dim=-1), 1)
                _, label = torch.max(label, 1)

                predictions = predictions.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                self.metrics.on_batch_end(epoch, i, predictions, label)

            acc_, f1 = self.metrics.on_epoch_end(epoch)
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
