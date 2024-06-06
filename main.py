import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import argparse
import pickle
parser = argparse.ArgumentParser()
import pandas as pd

from SSR import SSRModel 


HYPHEN='hyphen'
HYPHAT='hyphat'
SSR='ssr'
LORENTZ = 'lorentz'
POINCARE = 'poincare'
EUCLID = 'euclid'
POLITIFACT = 'politifact'
ANTIVAX= 'antivax'
GOSSIPCOP = 'gossipcop'
FIGLANG_TWITTER = 'figlang_twitter'
FIGLANG_REDDIT = 'figlang_reddit'
RUMOUREVAL = 'rumoureval'
TWITTER15 = 'twitter15'
TWITTER16 = 'twitter16'
PHEME = 'pheme'

parser.add_argument('--dataset', default= POLITIFACT, choices = [ANTIVAX, POLITIFACT, GOSSIPCOP, FIGLANG_REDDIT, FIGLANG_TWITTER, RUMOUREVAL, TWITTER15, TWITTER16, PHEME], help='Specify the dataset for which you want to run the experiments.')

# file = open(f'data/{args.dataset}/{args.dataset}_preprocessed.pkl', 'rb')
if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv(f'data/{args.dataset}/{args.dataset}.csv') 
    print(df)
    print(df['comment'])
    





