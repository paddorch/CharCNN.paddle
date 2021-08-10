from time import perf_counter

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack
from tqdm import tqdm

from char_cnn.models import Tokenizer, CharCNN


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
            collate_fn=collate, return_list=True,
            num_workers=16
        )

        return data_loader

    dataset = 'imdb'
    max_len = 2000
    batch_size = 500
    num_epochs = 100
    lr = 0.001

    train_dataset = load_dataset(dataset, splits='train')
    test_dataset = load_dataset(dataset, splits='test')

    tokenizer = Tokenizer(max_len=max_len)
    train_loader = create_dataloader(train_dataset, tokenizer=tokenizer, shuffle=True, batch_size=batch_size)
    test_loader = create_dataloader(test_dataset, tokenizer=tokenizer, shuffle=False, batch_size=batch_size)

    model = CharCNN(
        max_length=max_len, vocab_size=tokenizer.alphabet_size, num_classes=len(train_dataset.label_list)
    )

    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    criterion = paddle.nn.CrossEntropyLoss()

    def accuracy(output, target):
        prediction = paddle.argmax(output, axis=1)

        total = output.shape[0]
        correct = paddle.sum(prediction == target).item()

        return total, correct

    def evaluate(epoch: int):
        model.eval()

        total = 0
        correct = 0
        pbar = tqdm(total=len(test_dataset), desc=f'| E | epoch {epoch:03d}')
        for data in test_loader:
            input_ids, label = data

            output = model(input_ids)

            batch_total, batch_correct = accuracy(output, label)

            total += batch_total
            correct += batch_correct

            pbar.update(batch_total)
        pbar.close()

        return correct / total

    def train_step(epoch: int):
        model.train()
        optimizer.clear_grad()

        epoch_loss = 0.0
        total = 0
        correct = 0

        pbar = tqdm(total=len(train_dataset), desc=f'| T | epoch {epoch:03d}')
        for data in train_loader:
            input_ids, label = data

            output = model(input_ids)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            batch_total, batch_correct = accuracy(output, label)
            total += batch_total
            correct += batch_correct
            epoch_loss += loss.item()

            pbar.update(batch_total)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.close()

        return epoch_loss, correct / total

    for epoch in range(1, num_epochs + 1):
        tic = perf_counter()
        loss, acc = train_step(epoch)
        toc = perf_counter()
        print(f'| T | epoch {epoch}, loss {loss:.4f}, train acc {acc:.4f}, time {toc - tic:.4f}')

        tic = perf_counter()
        eval_acc = evaluate(epoch)
        toc = perf_counter()
        print(f'| E | epoch {epoch}, eval acc {eval_acc:.4f}, time {toc - tic:.4f}')



if __name__ == '__main__':
    main()
