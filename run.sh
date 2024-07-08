for model in ssm4rc 
do
    for embedding_dim in 100 
    do 
        for dataset in twitter15 twitter16 politifact antivax pheme figlang_twitter 
        do
            for manifold in lorentz euclidean 
            do
                CUDA_VISIBLE_DEVICES=0 python run.py --enable-log --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 30 --max-coms 30 --max-com-len 20 --max-sent-len 20 --log-path logging/run --model $model --embedding-dim $embedding_dim 
                CUDA_VISIBLE_DEVICES=1 python run.py --enable-log --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 32 --epochs 50 --max-sents 30 --max-coms 30 --max-com-len 20 --max-sent-len 20 --log-path logging/run --model $model --embedding-dim $embedding_dim --no-fourier 
            done
        done
    done
done
