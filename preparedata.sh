python amr/amr_gen.py --dataset $1 --max-comments 50
python amr/amr_var.py --dataset $1
python amr/amr_coref/amr_coref.py --dataset $1
python amr/amr_dummy.py --dataset $1
python amr/amr_dgl.py --dataset $1 --test-split 0.1
python preprocess.py --dataset $1
