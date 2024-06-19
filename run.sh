# for dataset in twitter15 twitter16 rumoureval pheme

for model in ssm4rc hyphen
do
    for embedding_dim in 200 100 50
    do 
        for dataset in politifact twitter16 twitter15 
        do
            for manifold in lorentz poincare euclid
            do
                CUDA_VISIBLE_DEVICES=0 python run.py --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 50 --max-coms 50 --max-com-len 10 --max-sent-len 50 --log-path logging/run --model $model --embedding-dim $embedding_dim  
            done
        done
    done
done
