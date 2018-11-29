from __future__ import print_function
import argparse
import json
import random
from util import Dictionary
import spacy
import csv
import tqdm

# lines = open(args.input).readlines()
# random.shuffle(lines)
# for i, line in enumerate(lines):
# item = json.loads(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenizer')
    parser.add_argument('--input', type=str, default='', help='input file')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument('--dict', type=str, default='', help='dictionary file')
    args = parser.parse_args()
    tokenizer = spacy.load('en')
    dictionary = Dictionary()
    dictionary.add_word('<pad>')  # add padding word
    with open(args.output, 'w') as fout:
        pool = []
        with open(args.input) as csvfile:
            readCSV = csv.DictReader(csvfile, delimiter=',')
            for item in readCSV:
                pool.append(item)

        random.shuffle(pool)

        numIters = 1000#len(pool)
        qdar = tqdm.tqdm(range(numIters),total= numIters,ascii=True)
        for i in qdar:
            # for item in pool:
            item = pool[i]
            words = tokenizer(' '.join(item['text'].split()))
            data = {
                'label': int(item['stars']) - 1,
                'text': list(map(lambda x: x.text.lower(), words))
            }
            fout.write(json.dumps(data) + '\n')
            for item in data['text']:
                dictionary.add_word(item)
            qdar.set_postfix(dictSize=str(len(dictionary)))
            # if i % 100 == 99:
            #     print('%d/%d files done, dictionary size: %d' %
            #           (i + 1, len(pool), len(dictionary)))
        fout.close()

    with open(args.dict, 'w') as fout:  # save dictionary for fast next process
        fout.write(json.dumps(dictionary.idx2word) + '\n')
        fout.close()
