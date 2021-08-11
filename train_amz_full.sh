python train.py \
  --train_path data/amazon_review_full_csv/train.csv.split_train \
  --val_path data/amazon_review_full_csv/train.csv.split_val \
#  --train_path /home/xuyichen/.paddlenlp/datasets/AmazonReviewFull/amazon_review_full_csv/train.csv \
#  --val_path /home/xuyichen/.paddlenlp/datasets/AmazonReviewFull/amazon_review_full_csv/test.csv \
  --save_folder output/models_amz_full \
  --batch_size 64 \
#  --device 3 \
  --val_interval 2000 \
  --data_augment True
