# Training time per Batch size
# python train_attack.py --device cuda:[ID] --checkpoint cifar10_resnet18_1.0 --tag batch_time/001 -c cifar10-optim/adam --epochs    4 --batch-size   1


# python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 4 --hs 10 --nl 10 --tag bs4_hs10_nl10

# python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 16 --hs 10 --nl 10 --tag bs16_hs10_nl10

python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 4 --hs 100 --nl 10 --tag bs4_hs100_nl10

python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 16 --hs 100 --nl 10 --tag bs16_hs100_nl10

python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 4 --hs 100 --nl 100 --tag bs32_hs100_nl100

python ./script/train.py --seed 333 --model rnn --epoch 100 --batch-size 16 --hs 100 --nl 100 --tag bs64_hs100_nl100



