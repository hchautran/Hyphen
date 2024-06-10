import os
import pandas as pd
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import argparse
import pickle
parser = argparse.ArgumentParser()
from trainer import Trainer 
from utils.dataset import FakeNewsDataset
import dgl
from const import *


parser.add_argument('--manifold', choices=[POINCARE, EUCLID, LORENTZ], default = 'PoincareBall', help='Choose the underlying manifold for Hyphen')
parser.add_argument('--no-fourier', default=True, action='store_false', help='If you want to remove the Fourier sublayer from Hyphen\'s co-attention module.')
parser.add_argument('--no-comment', default=True, action='store_false', help='If you want to remove the comment module from Hyphen i.e. just consider news content as the only input modality.')
parser.add_argument('--no-content', default=True, action='store_false', help='If you want to remove the content module from Hyphen, i.e. just consider user comments as the only input modality.')
parser.add_argument('--log-path', type = str, default = "logging/run", help='Specify the path of the Log file for Tensorboard.')
parser.add_argument('--lr', default = 0.001, type = float, help='Specify the learning rate for Hyphen.')
parser.add_argument('--dataset', default= POLITIFACT, choices = [ANTIVAX, POLITIFACT, GOSSIPCOP, FIGLANG_REDDIT, FIGLANG_TWITTER, RUMOUREVAL, TWITTER15, TWITTER16, PHEME], help='Specify the dataset for which you want to run the experiments.')
parser.add_argument('--max-coms', type = int, default= 10, help='Specify the maximum number of user comments (per post) you want to consider.')
parser.add_argument('--max-sents', type = int, default= 20, help='Specify the maximum number of news sentences from the social media post that you want to consider.')
parser.add_argument('--max-com-len', type = int, default= 10, help='Specify the maximum length of a user comment to feed in Hyphen.')
parser.add_argument('--max-sent-len', type = int, default = 10, help='Specify the maximum length of a news sentence.')
parser.add_argument('--batch-size', type = int,  default = 32,  help='Specify the batch size of the dataset.')
parser.add_argument('--epochs', type = int, default= 5, help='The number of epochs to train Hyphen.')
parser.add_argument('--use_gat', default=False, action='store_true', help='use graph attention network')
parser.add_argument('--enable-log', default=False, action='store_true', help='use graph attention network')
parser.add_argument('--model', default=HYPHEN)
args = parser.parse_args()

file = open(f'data/{args.dataset}/{args.dataset}_preprocessed.pkl', 'rb')
df = pd.read_csv(f'data/{args.dataset}/{args.dataset}.csv') 
props = pickle.load(file)

id_train, id_test = props['train']['id'], props['val']['id']
x_train, x_val = props['train']['x'], props['val']['x']
y_train, y_val = props['train']['y'], props['val']['y']
c_train, c_val = props['train']['c'], props['val']['c']
raw_c_train, raw_c_val = list(df[df['id'].isin(id_train)]['comments']), list(df[df['id'].isin(id_test)]['comments'])
sub_train, sub_val = props['train']['subgraphs'], props['val']['subgraphs']


c_train = [dgl.add_self_loop(i) for i in c_train]
c_val = [dgl.add_self_loop(i) for i in c_val]
print("Building model....")

print("Model built.")

print("Encoding texts....")
# Create encoded input for content and comments

print("preparing dataset....")
train_dataset = FakeNewsDataset(
    x_train,
    c_train,
    y_train,
    sub_train,
    args.max_com_length,
    args.max_sent_length,

)
val_dataset = FakeNewsDataset(
    x_val,
    c_val,
    y_val,
    sub_val,
    args.max_com_length,
    args.max_sent_length,
)


trainer = Trainer(
    n_classes=train_dataset.num_classes,
    max_com_len=args.max_com_length,
    max_com_len=args.max_com_length,
    batch_size=args.batch_size
)
trainer.train(epochs=args.epochs)