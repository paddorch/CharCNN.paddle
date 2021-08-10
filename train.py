import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack

from char_cnn.models.tokenizer import Tokenizer


def main():
    def create_dataloader(dataset, tokenizer: Tokenizer, shuffle: bool, batch_size: int):
        def collate(datalist, stack_fn=Stack(dtype='int64')):
            input_ids = [tokenizer.tokenize(data['text']) for data in datalist]
            labels = [data['label'] for data in datalist]

            input_ids = stack_fn(input_ids)
            labels = stack_fn(labels)

            return [input_ids, labels]

        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        data_loader = paddle.io.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate, return_list=True
        )

        return data_loader

    dataset = 'imdb'
    max_len = 2000
    batch_size = 32

    train_dataset = load_dataset(dataset, splits='train')
    test_dataset = load_dataset(dataset, splits='test')

    tokenizer = Tokenizer(max_len=max_len)
    train_loader = create_dataloader(train_dataset, tokenizer=tokenizer, shuffle=True, batch_size=batch_size)
    test_loader = create_dataloader(test_dataset, tokenizer=tokenizer, shuffle=False, batch_size=batch_size)

    for data in train_loader:
        print(data)
        break

    pass


if __name__ == '__main__':
    main()
