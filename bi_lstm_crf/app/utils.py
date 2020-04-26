from os.path import exists, join
import torch
from bi_lstm_crf.model import BiRnnCrf

FILE_ARGUMENTS = "arguments.json"
FILE_MODEL = "model.pth"


def arguments_filepath(model_dir):
    return join(model_dir, FILE_ARGUMENTS)


def model_filepath(model_dir):
    return join(model_dir, FILE_MODEL)


def build_model(args, processor, load=True, verbose=False):
    model = BiRnnCrf(len(processor.vocab), len(processor.tags),
                     embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_rnn_layers=args.num_rnn_layers)

    # weights
    model_path = model_filepath(args.model_dir)
    if exists(model_path) and load:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if verbose:
            print("load model weights from {}".format(model_path))
    return model


def running_device(device):
    return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
