python train.py \
  --train_path /hy-tmp/datasets/yahoo_answers_csv/train.csv.split_train \
  --val_path /hy-tmp/datasets/yahoo_answers_csv/train.csv.split_dev \
  --save_folder output/models_yahoo_answers\
  --batch_size 64 \
  --val_interval 2000
  --data_augment True