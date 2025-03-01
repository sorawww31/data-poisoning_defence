mkdir -p log/imagenet/ResNet34/neutral/
mkdir -p log/imagenet/ResNet34/wolfe/
mkdir -p poisons/imagenet/ResNet34/gradmatch/
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1000000000 --modelkey 1000000000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1000000000.pt >& log/imagenet/ResNet34/neutral/neutral_1000000000.log
python brew_poison.py --wolfe 0.9 1e-4 --linesearch_epoch 60 --net ResNet34 --name 3_top_right_wolfe --poisonkey 1000000000 --modelkey 1000000000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1100000000.pt --load_poison poisons/imagenet/ResNet34/gradmatch/1000000000.pt >& log/imagenet/ResNet34/wolfe/wolfe_1000000000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1100000000 --modelkey 1100000000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison --wandb poisons/imagenet/ResNet34/gradmatch/1100000000.pt >& log/imagenet/ResNet34/neutral/neutral_1100000000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1110000000 --modelkey 1110000000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1110000000.pt >& log/imagenet/ResNet34/neutral/neutral_1110000000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111000000 --modelkey 1111000000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111000000.pt >& log/imagenet/ResNet34/neutral/neutral_1111000000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111100000 --modelkey 1111100000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111100000.pt >& log/imagenet/ResNet34/neutral/neutral_1111100000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111110000 --modelkey 1111110000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111110000.pt >& log/imagenet/ResNet34/neutral/neutral_1111110000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111111000 --modelkey 1111111000 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111111000.pt >& log/imagenet/ResNet34/neutral/neutral_1111111000.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111111100 --modelkey 1111111100 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111111100.pt >& log/imagenet/ResNet34/neutral/neutral_1111111100.log 
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111111110 --modelkey 1111111110 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111111110.pt >& log/imagenet/ResNet34/neutral/neutral_1111111110.log
python brew_poison.py --net ResNet34 --name 3_top_right_neutral --poisonkey 1111111111 --modelkey 1111111111 --pretrained --restart 8 --vruns 1 --dataset ImageNet --optimization imagenet --data_path /workspace/imagenet --pbatch 128 --eps 16 --budget 0.001 --save_poison poisons/imagenet/ResNet34/gradmatch/1111111111.pt >& log/imagenet/ResNet34/neutral/neutral_1111111111.log