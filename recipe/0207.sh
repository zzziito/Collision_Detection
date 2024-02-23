# RNN

python ./script/train.py --seed 333 --model rnn --epoch 2 --batch-size 64 --learning-rate 1.0E-6 --hidden-size 256 --num-layers 20 --tag hs256_nl20_bs64

python ./script/train.py --seed 333 --model rnn --epoch 2 --batch-size 128 --learning-rate 1.0E-6 --hidden-size 256 --num-layers 20 --tag hs256_nl20

python ./script/train.py --seed 333 --model rnn --epoch 2 --batch-size 128 --learning-rate 1.0E-6 --hidden-size 512 --num-layers 40 --tag hs512_nl40

python ./script/train.py --seed 333 --model rnn --epoch 2 --batch-size 128 --learning-rate 1.0E-6 --hidden-size 512 --num-layers 40 --tag hs512_nl40


# Transformer

# python ./script/train.py --seed 333 --model transformer --epoch 2 --batch-size 4 --hidden-size 512 --learning-rate 1.0E-6 --nhead 16 --nel 6 --tag nhead16_nel6

# python ./script/train.py --seed 333 --model transformer --epoch 2 --batch-size 4 --hidden-size 512 --learning-rate 1.0E-6 --nhead 4 --nel 6 --tag nhead4_nel6

# python ./script/train.py --seed 333 --model transformer --epoch 2 --batch-size 4 --learning-rate 0.0001 --nhead 4 --nel 6 --tag bs4_nhead4_nel6_lr4_epoch200

# python ./script/train.py --seed 333 --model transformer --epoch 2 --batch-size 4 --learning-rate 0.00001 --nhead 8 --nel 6 --tag bs4_nhead8_nel6_lr5



# python ./script/train_disc.py --seed 333 --model rnn --epoch 2 --batch-size 4 --learning-rate 0.0001 --hidden-size 100 --num-layers 10 --tag bs4_hs100_nl10_lr4
