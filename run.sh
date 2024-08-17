for manifold in euclid lorentz poincare
do
    for model in hs4 
    do
        for embedding_dim in 200 
        do 
            for dataset in twitter15 twitter16 politifact pheme figlang_twitter 
            do
                CUDA_VISIBLE_DEVICES=$1 python run.py --manifold $manifold --lr 0.001 --dataset $dataset --batch-size 30 --epochs 100 --max-sents 20 --max-coms 20 --max-com-len 20 --max-sent-len 20 --log-path logging/run --model $model --embedding-dim $embedding_dim --enable-log --no-fourier  
                # CUDA_VISIBLE_DEVICES=$1 python run.py --enable-log --manifold $2 --lr 0.001 --dataset $dataset --batch-size 32 --epochs 100 --max-sents 30 --max-coms 30 --max-com-len 20 --max-sent-len 20 --log-path logging/run --model $model --embedding-dim $embedding_dim --no-fourier 
            done
        done
    done
done
