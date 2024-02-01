# Training time per Batch size
# python train_attack.py --device cuda:[ID] --checkpoint cifar10_resnet18_1.0 --tag batch_time/001 -c cifar10-optim/adam --epocnhead    4 --batch-size   1


python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 4 --nhead 8 --nel 6 --tag bs4_nhead8_nel6

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 16 --nhead 8 --nel 6 --tag bs16_nhead8_nel6

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 4 --nhead 8 --nel 8 --tag bs4_nhead8_nel8

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 16 --nhead 8 --nel 8 --tag bs16_nhead8_nel8

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 4 --nhead 8 --nel 10 --tag bs4_nhead8_nel10

python ./script/train.py --seed 333 --model transformer --epoch 100 --batch-size 16 --nhead 8 --nel 10 --tag bs16_nhead8_nel10



