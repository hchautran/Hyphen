# for dataset in twitter15 twitter16 rumoureval pheme

for embedding_dim in 50 100 
do 
    for dataset in twitter15 twitter16 antivax politifact  pheme  
    do
        # for model in  ssm4rc hyphen
        # do
            CUDA_VISIBLE_DEVICES=$1 python run.py --manifold poincare --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 50 --max-coms 50 --max-com-len 10 --max-sent-len 50 --log-path logging/run --model $2 --embedding-dim $embedding_dim --enable-log
        # done
    done
done
