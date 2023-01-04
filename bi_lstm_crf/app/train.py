from os import mkdir
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from bi_lstm_crf.app.preprocessing import *
from bi_lstm_crf.app.utils import *


def __eval_model(model, device, dataloader, desc):
    model.eval()
    with torch.no_grad():
        # eval
        losses, nums = zip(*[
            (model.loss(xb.to(device), yb.to(device)), len(xb))
            for xb, yb in tqdm(dataloader, desc=desc)])
        return torch.sum(torch.multiply(torch.tensor(losses), torch.tensor(nums))) / np.sum(nums)


def __save_loss(losses, file_path):
    pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv(file_path, index=False)


def __save_model(model_dir, model):
    model_path = model_filepath(model_dir)
    torch.save(model.state_dict(), model_path)
    print("save model => {}".format(model_path))


def train(args):
    model_dir = args.model_dir
    if not exists(model_dir):
        mkdir(model_dir)
    save_json_file(vars(args), arguments_filepath(model_dir))

    preprocessor = Preprocessor(config_dir=args.corpus_dir, save_config_dir=args.model_dir, verbose=True)
    model = build_model(args, preprocessor, load=args.recovery, verbose=True)

    # loss
    loss_path = join(args.model_dir, "loss.csv")
    losses = pd.read_csv(loss_path).values.tolist() if args.recovery and exists(loss_path) else []

    # datasets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocessor.load_dataset(
        args.corpus_dir, args.val_split, args.test_split, max_seq_len=args.max_seq_len)
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    valid_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size * 2)
    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size * 2)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    device = running_device(args.device)
    model.to(device)

    val_loss = 0
    best_val_loss = 1e4
    for epoch in range(args.num_epoch):
        # train
        model.train()
        bar = tqdm(train_dl)
        for bi, (xb, yb) in enumerate(bar):
            model.zero_grad()

            loss = model.loss(xb.to(device), yb.to(device))
            loss.backward()
            optimizer.step()
            bar.set_description("{:2d}/{} loss: {:5.2f}, val_loss: {:5.2f}".format(
                epoch+1, args.num_epoch, loss, val_loss))
            losses.append([epoch, bi, loss.item(), np.nan])

        # evaluation
        val_loss = __eval_model(model, device, dataloader=valid_dl, desc="eval").item()
        # save losses
        losses[-1][-1] = val_loss
        __save_loss(losses, loss_path)

        # save model
        if not args.save_best_val_model or val_loss < best_val_loss:
            best_val_loss = val_loss
            __save_model(args.model_dir, model)
            print("save model(epoch: {}) => {}".format(epoch, loss_path))

    # test
    test_loss = __eval_model(model, device, dataloader=test_dl, desc="test").item()
    last_loss = losses[-1][:]
    last_loss[-1] = test_loss
    losses.append(last_loss)
    __save_loss(losses, loss_path)
    print("training completed. test loss: {:.2f}".format(test_loss))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--model_dir', type=str, default="model_dir", help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true", help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    train(args)


if __name__ == "__main__":
    main()
