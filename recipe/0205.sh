# experiment on new dataset

# RNN

# python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 4 --learning-rate 0.0001 --hidden-size 100 --num-layers 10 --tag bs4_hs100_nl10_lr4

# python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 4 --learning-rate 0.001 --hidden-size 100 --num-layers 10 --tag bs4_hs100_nl10_lr3

# Transformer

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 4 --learning-rate 0.0001 --hidden-size 512 --nhead 8 --nel 6 --tag bs4_nhead8_nl10_lr4

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 4 --learning-rate 0.001 --hidden-size 512 --nhead 8 --nel 6 --tag bs4_nhead8_nl10_lr3
