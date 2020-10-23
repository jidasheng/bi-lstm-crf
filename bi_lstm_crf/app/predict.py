import argparse
import numpy as np
from bi_lstm_crf.app.preprocessing import *
from bi_lstm_crf.app.utils import *


class WordsTagger:
    def __init__(self, model_dir, device=None):
        args_ = load_json_file(arguments_filepath(model_dir))
        args = argparse.Namespace(**args_)
        args.model_dir = model_dir
        self.args = args

        self.preprocessor = Preprocessor(config_dir=model_dir, verbose=False)
        self.model = build_model(self.args, self.preprocessor, load=True, verbose=False)
        self.device = running_device(device)
        self.model.to(self.device)

        self.model.eval()

    def __call__(self, sentences, begin_tags="BS"):
        """predict texts

        :param sentences: a text or a list of text
        :param begin_tags: begin tags for the beginning of a span
        :return:
        """
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("sentences must be a list of sentence")

        try:
            sent_tensor = np.asarray([self.preprocessor.sent_to_vector(s) for s in sentences])
            sent_tensor = torch.from_numpy(sent_tensor).to(self.device)
            with torch.no_grad():
                _, tags = self.model(sent_tensor)
            tags = self.preprocessor.decode_tags(tags)
        except RuntimeError as e:
            print("*** runtime error: {}".format(e))
            raise e
        return tags, self.tokens_from_tags(sentences, tags, begin_tags=begin_tags)

    @staticmethod
    def tokens_from_tags(sentences, tags_list, begin_tags):
        """extract entities from tags

        :param sentences: a list of sentence
        :param tags_list: a list of tags
        :param begin_tags:
        :return:
        """
        if not tags_list:
            return []

        def _tokens(sentence, ts):
            # begins: [(idx, label), ...]
            all_begin_tags = begin_tags + "O"
            begins = [(idx, t[2:]) for idx, t in enumerate(ts) if t[0] in all_begin_tags]
            begins = [
                         (idx, label)
                         for idx, label in begins
                         if ts[idx] != "O" or (idx > 0 and ts[idx - 1] != "O")
                     ] + [(len(ts), "")]

            tokens_ = [(sentence[s:e], label) for (s, label), (e, _) in zip(begins[:-1], begins[1:]) if label]
            return [((t, tag) if tag else t) for t, tag in tokens_]

        tokens_list = [_tokens(sentence, ts) for sentence, ts in zip(sentences, tags_list)]
        return tokens_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str, help="the sentence to be predicted")
    parser.add_argument('--model_dir', type=str, required=True, help="the model directory for model files")
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')

    args = parser.parse_args()

    results = WordsTagger(args.model_dir, args.device)([args.sentence])
    print(args.sentence)
    for objs in results:
        print(json.dumps(objs[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
