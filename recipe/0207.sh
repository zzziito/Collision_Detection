# RNN

# python ./script/train.py --seed 333 --model rnn --epoch 20 --batch-size 4 --learning-rate 0.0001 --hidden-size 100 --num-layers 10 --tag bs4_hs100_nl10_lr4

# python ./script/train.py --seed 333 --model rnn --epoch 20 --batch-size 4 --learning-rate 0.0001 --hidden-size 50 --num-layers 10 --tag bs4_hs50_nl10_lr4

# python ./script/train.py --seed 333 --model rnn --epoch 20 --batch-size 4 --learning-rate 0.0001 --hidden-size 100 --num-layers 20 --tag bs4_hs100_nl20_lr4

# python ./script/train.py --seed 333 --model rnn --epoch 20 --batch-size 4 --learning-rate 0.00001 --hidden-size 100 --num-layers 10 --tag bs4_hs100_nl10_lr5


# Transformer

python ./script/train.py --seed 333 --model transformer --epoch 20 --batch-size 4 --learning-rate 0.0001 --nhead 8 --nel 6 --tag bs4_nhead8_nel6_lr4

python ./script/train.py --seed 333 --model transformer --epoch 20 --batch-size 4 --learning-rate 0.0001 --nhead 4 --nel 6 --tag bs4_nhead4_nel6_lr4

python ./script/train.py --seed 333 --model transformer --epoch 20 --batch-size 4 --learning-rate 0.0001 --nhead 4 --nel 6 --tag bs4_nhead4_nel6_lr4_epoch200

python ./script/train.py --seed 333 --model transformer --epoch 20 --batch-size 4 --learning-rate 0.00001 --nhead 8 --nel 6 --tag bs4_nhead8_nel6_lr5
