python train.py \
  --train_path /home/xuyichen/.paddlenlp/datasets/AmazonReviewFull/amazon_review_full_csv/train.csv.split_train \
  --val_path /home/xuyichen/.paddlenlp/datasets/AmazonReviewFull/amazon_review_full_csv/train.csv.split_val \
  --save_folder output/models_amz_full \
  --batch_size 64 \
  --gpu 3 \
  --val_interval 2000 \
  --data_augment True
