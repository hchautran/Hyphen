
# for dataset in pheme twitter16 twitter15 rumoureval 
# for dataset in  'antivax' 'gossipcop' 'figlang_twitter' 'figlang_reddit' 'Gossipcop'
for dataset in 'gossipcop'  'antivax' 'politifact' 'twitter15' 'twitter16' 'rumoureval' 'HASOC' 'figlang_reddit' 'figlang_twitter'
do
   CUDA_VISIBLE_DEVICES="0" python run.py --manifold $1 --lr 0.001 --dataset $dataset --batch-size 32 --epochs 100 --max-sents 50 --max-coms 50 --max-com-len 10 --max-sent-len 50 --log-path logging/run  --use_gat --model $2  --enable-log
done

