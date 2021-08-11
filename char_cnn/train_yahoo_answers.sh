python train.py \
  --train_path /home/xuyichen/.paddlenlp/datasets/YahooAnswers/yahoo_answers_csv/train.csv.split_train \
  --val_path /home/xuyichen/.paddlenlp/datasets/YahooAnswers/yahoo_answers_csv/train.csv.split_val \
  --save_folder output/models_yahoo_answers \
  --batch_size 64 \
  --gpu 2 \
  --val_interval 2000 \
  --data_augment True
