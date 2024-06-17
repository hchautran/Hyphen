# for dataset in twitter15 twitter16 rumoureval pheme

for embedding_dim in 200 100 50
do 
    for dataset in politifact twitter15 twitter16 
    do
        for manifold in euclid lorentz poincare
        do
        # for model in  ssm4rc hyphen
        # do
            # CUDA_VISIBLE_DEVICES=1 python run.py --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 50 --max-coms 50 --max-com-len 10 --max-sent-len 50 --log-path logging/run --model hyphen --embedding-dim $embedding_dim  --enable-log
            CUDA_VISIBLE_DEVICES=1 python run.py --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 50 --max-coms 50 --max-com-len 10 --max-sent-len 50 --log-path logging/run --model ssm4rc --embedding-dim $embedding_dim  --enable-log
        # done
        done
    done
done
