for dataset in politifact twitter16 twitter15 rumoureval pheme
do
    python run.py --manifold $1 --lr 0.001 --dataset $dataset --batch-size 32 --epochs 100 --max-sents 20 --max-coms 10 --max-com-len 10 --max-sent-len 20 --log-path logging/run --enable-log --use_gat
done
