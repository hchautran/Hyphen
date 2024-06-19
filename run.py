import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import argparse
parser = argparse.ArgumentParser()
# from LorentzTrainer import Trainer as LorentzTrainer
from trainer import Trainer 
from const import * 
import pandas as pd

parser.add_argument('--manifold', choices=[EUCLID, LORENTZ, POINCARE], default = POINCARE, help='Choose the underlying manifold for Hyphen')
parser.add_argument('--no-fourier', default=True, action='store_false', help='If you want to remove the Fourier sublayer from Hyphen\'s co-attention module.')
parser.add_argument('--no-comment', default=True, action='store_false', help='If you want to remove the comment module from Hyphen i.e. just consider news content as the only input modality.')
parser.add_argument('--no-content', default=True, action='store_false', help='If you want to remove the content module from Hyphen, i.e. just consider user comments as the only input modality.')
parser.add_argument('--log-path', type = str, default = "logging/run", help='Specify the path of the Log file for Tensorboard.')
parser.add_argument('--lr', default = 0.001, type = float, help='Specify the learning rate for Hyphen.')
parser.add_argument('--dataset', default= 'politifact', choices = ['antivax', 'politifact', 'gossipcop', 'figlang_twitter', 'figlang_reddit', 'twitter16', 'rumoureval', 'pheme', 'twitter15', 'hasoc'], help='Specify the dataset for which you want to run the experiments.')
parser.add_argument('--max-coms', type = int, default= 10, help='Specify the maximum number of user comments (per post) you want to consider.')
parser.add_argument('--max-sents', type = int, default= 20, help='Specify the maximum number of news sentences from the social media post that you want to consider.')
parser.add_argument('--max-com-len', type = int, default= 10, help='Specify the maximum length of a user comment to feed in Hyphen.')
parser.add_argument('--max-sent-len', type = int, default = 10, help='Specify the maximum length of a news sentence.')
parser.add_argument('--batch-size', type = int,  default = 32,  help='Specify the batch size of the dataset.')
parser.add_argument('--epochs', type = int, default= 100, help='The number of epochs to train Hyphen.')
parser.add_argument('--model', type = str, default= HYPHEN, help='model type')
parser.add_argument('--enable-log', action='store_true', default=False , help='log to wandb')
parser.add_argument('--embedding-dim', default=100, help='embedding dim')

args = parser.parse_args()

file = open(f'{DATA_PATH}/data/{args.dataset}/{args.dataset}_preprocessed.pkl', 'rb')
df = pd.read_csv(f'data/{args.dataset}/{args.dataset}.csv') 
props = pickle.load(file)

id_train, id_test = props['train']['id'], props['val']['id']
raw_c_train, raw_c_val = list(df[df['id'].isin(id_train)]['comments']), list(df[df['id'].isin(id_test)]['comments'])
raw_c_train = [c.split('::') if isinstance(c, str) else '' for c in raw_c_train]
raw_c_val= [c.split('::') if isinstance(c, str) else '' for c in raw_c_val]
x_train, x_val = props['train']['x'], props['val']['x']
y_train, y_val = props['train']['y'], props['val']['y']
c_train, c_val = props['train']['c'], props['val']['c']
sub_train, sub_val = props['train']['subgraphs'], props['val']['subgraphs']

model = Trainer(
    manifold=args.manifold,
    model_type=args.model,
    platform=args.dataset, 
    max_sen_len=args.max_sent_len, 
    max_com_len=args.max_com_len, 
    max_sents=args.max_sents, 
    max_coms=args.max_coms, 
    lr = args.lr, 
    fourier = args.no_fourier,
    curv=1.0,
    enable_log=args.enable_log,
    embedding_dim=int(args.embedding_dim)
)


model.train(
    train_x=x_train, 
    train_y=y_train, 
    train_c=c_train, 
    train_raw_c=raw_c_train, 
    val_x=x_val, 
    val_y=y_val, 
    val_c=c_val, 
    val_raw_c=raw_c_val, 
    sub_train=sub_train, 
    sub_val=sub_val, 
    batch_size=args.batch_size, 
    epochs=args.epochs
)
