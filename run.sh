for model in ssm4rc hyphen
do
    for embedding_dim in 200 100 50 
    do 
        for dataset in politifact twitter15 twitter16 antivax pheme figlang_twitter rumoureval 
        do
            for manifold in lorentz poincare euclid
            do
                CUDA_VISIBLE_DEVICES=1 python run.py --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 30 --max-coms 30 --max-com-len 20 --max-sent-len 20 --log-path logging/run --model $model --embedding-dim $embedding_dim --enable-log 
                CUDA_VISIBLE_DEVICES=1 python run.py --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 30 --max-coms 30 --max-com-len 20 --max-sent-len 20 --log-path logging/run --model $model --embedding-dim $embedding_dim --no-fourier --enable-log
            done
        done
    done
done
