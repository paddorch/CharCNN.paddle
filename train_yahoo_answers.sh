python train.py \
  --train_path /home/xuyichen/.paddlenlp/datasets/YahooAnswers/yahoo_answers_csv/train.csv \
  --val_path /home/xuyichen/.paddlenlp/datasets/YahooAnswers/yahoo_answers_csv/test.csv \
  --save_folder output/models_yahoo_answers_2 \
  --batch_size 64 \
  --gpu 3 \
  --val_interval 2000 \
  --data_augment True \
  --continue_from output/models_yahoo_answers/CharCNN_best.pth.tar

#  --train_path /hy-tmp/datasets/yahoo_answers_csv/train.csv.split_train \
#  --val_path /hy-tmp/datasets/yahoo_answers_csv/train.csv.split_dev \
