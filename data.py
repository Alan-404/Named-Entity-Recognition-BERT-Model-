from preprocessing.data import DataProcessor
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--clean_path", type=str)

    args = parser.parse_args()

    processor = DataProcessor(args.tokenizer_path)

    words, tags = processor.process(args.data_path, data_path=args.clean_path)

    print(words.shape)
    print(tags.shape)

    print(f"Dictionary Size: {len(processor.dictionary)}")
    print(f"Number of Entities: {len(processor.entities)}")