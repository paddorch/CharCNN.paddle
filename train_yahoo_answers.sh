python train.py \
  --train_path data/yahoo_answers_csv/train.csv.split_train \
  --val_path data/yahoo_answers_csv/train.csv.split_dev \
#  --train_path /home/xuyichen/.paddlenlp/datasets/YahooAnswers/yahoo_answers_csv/train.csv \
#  --val_path /home/xuyichen/.paddlenlp/datasets/YahooAnswers/yahoo_answers_csv/test.csv \
  --save_folder output/models_yahoo_answers \
  --batch_size 64 \
#  --device 2 \
  --val_interval 2000 \
  --data_augment True
