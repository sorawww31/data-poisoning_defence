mkdir -p poisons/vgg11/gradmatch/
mkdir -p log/transfer/

#python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 1 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/1.pt --only_brew > log/transfer/vgg_to_resnet1.log 2>&1
#python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 1 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/1.pt >>  log/transfer/vgg_to_resnet1.log 2>&1

#python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 2 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/2.pt --only_brew > log/transfer/vgg_to_resnet2.log 2>&1
#python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 2 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/2.pt >>  log/transfer/vgg_to_resnet2.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 3 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/3.pt --only_brew > log/transfer/vgg_to_resnet3.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 3 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/3.pt >>  log/transfer/vgg_to_resnet3.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 4 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/4.pt --only_brew > log/transfer/vgg_to_resnet4.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 4 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/4.pt >>  log/transfer/vgg_to_resnet4.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 5 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/5.pt --only_brew > log/transfer/vgg_to_resnet5.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 5 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/5.pt >>  log/transfer/vgg_to_resnet5.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 6 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/6.pt --only_brew > log/transfer/vgg_to_resnet6.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 6 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/6.pt >>  log/transfer/vgg_to_resnet6.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 7 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/7.pt --only_brew > log/transfer/vgg_to_resnet7.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 7 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/7.pt >>  log/transfer/vgg_to_resnet7.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 8 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/8.pt --only_brew > log/transfer/vgg_to_resnet8.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 8 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/8.pt >>  log/transfer/vgg_to_resnet8.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 9 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/9.pt --only_brew > log/transfer/vgg_to_resnet9.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 9 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/9.pt >>  log/transfer/vgg_to_resnet9.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 10 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/vgg11/gradmatch/10.pt --only_brew > log/transfer/vgg_to_resnet10.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 10 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/vgg11/gradmatch/10.pt >>  log/transfer/vgg_to_resnet10.log 2>&1