mkdir -p cifar10/vgg16/gradmatch 
mkdir -p log/transfer
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 1 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/1.pt --only_brew >log/transfer/resnet_to_vgg1.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 1 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/1.pt >> log/transfer/resnet_to_vgg1.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 2 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/2.pt --only_brew >log/transfer/resnet_to_vgg2.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 2 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/2.pt >> log/transfer/resnet_to_vgg2.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 3 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/3.pt --only_brew >log/transfer/resnet_to_vgg3.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 3 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/3.pt >> log/transfer/resnet_to_vgg3.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 4 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/4.pt --only_brew >log/transfer/resnet_to_vgg4.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 4 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/4.pt >> log/transfer/resnet_to_vgg4.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 5 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/5.pt --only_brew >log/transfer/resnet_to_vgg5.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 5 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/5.pt >> log/transfer/resnet_to_vgg5.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 6 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/6.pt --only_brew >log/transfer/resnet_to_vgg6.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 6 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/6.pt >> log/transfer/resnet_to_vgg6.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 7 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/7.pt --only_brew >log/transfer/resnet_to_vgg7.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 7 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/7.pt >> log/transfer/resnet_to_vgg7.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 8 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/8.pt --only_brew >log/transfer/resnet_to_vgg8.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 8 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/8.pt >> log/transfer/resnet_to_vgg8.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 9 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/9.pt --only_brew >log/transfer/resnet_to_vgg9.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 9 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/9.pt >> log/transfer/resnet_to_vgg9.log 2>&1

python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net ResNet18 --name poisonvgg_resnet --poisonkey 10 --restart 8 --vruns 1 --pbatch 512 --eps 16 --budget 0.01  --save_poison poisons/cifar10/vgg16/gradmatch/10.pt --only_brew >log/transfer/resnet_to_vgg10.log 2>&1
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 25 --net VGG11 --name poisonvgg_resnet --poisonkey 10 --restart 8 --vruns 5 --pbatch 512 --eps 16 --budget 0.01  --load_poison poisons/cifar10/vgg16/gradmatch/10.pt >> log/transfer/resnet_to_vgg10.log 2>&1