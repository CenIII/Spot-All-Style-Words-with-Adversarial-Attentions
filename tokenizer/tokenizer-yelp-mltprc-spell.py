from __future__ import print_function
import argparse
import json
import random
from util import Dictionary
import spacy
import csv
import tqdm

import os

from symspellpy.symspellpy import SymSpell, Verbosity  # import the module
from multiprocessing import Process


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenizer')
    parser.add_argument('--input', type=str, default='', help='input file')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument('--dict', type=str, default='', help='dictionary file')
    args = parser.parse_args()
    tokenizer = spacy.load('en')
    numProc = 4
    # dictionary = Dictionary()
    # dictionary.add_word('<pad>')  # add padding word

    # init for symspell
    # create object
    initial_capacity = 83000
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,
                         prefix_length)
    # load dictionary
    dictionary_path = os.path.join(os.path.dirname(__file__),
                                   "frequency_dictionary_en_82_765.txt")
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")

    def SymSpellCheck(text):
        # input_term = ("whereis th elove hehad dated forImuch of thepast who "
                  # "couqdn'tread in sixtgrade and ins pired him")
        input_term = (text)
        # max edit distance per lookup (per single word, not per whole input string)
        max_edit_distance_lookup = 2
        suggestions = sym_spell.lookup_compound(input_term,
                                                max_edit_distance_lookup)
        # display suggestion term, edit distance, and term frequency
        # for suggestion in suggestions:
        #     print("{}, {}, {}".format(suggestion.term, suggestion.count,
        #                               suggestion.distance))
        return suggestions[0].term.split(' ')

    def proc_token(tk):
        if tk.replace('.','',1).isdigit():
            return "_num_"
        return tk

    def sub_process(numIters,pool,id):
        dictionary = Dictionary()
        dictionary.add_word('<pad>')  # add padding word
        with open(args.output+str(id), 'w') as fout:
            qdar = tqdm.tqdm(range(numIters),total= numIters,ascii=True)
            for i in qdar:
                # for item in pool:
                item = pool[i]


                # words = tokenizer(' '.join(item['text'].split()))
                words = SymSpellCheck(item['text'])


                data = {
                    'label': int(item['stars']) - 1,
                    'text': list(map(lambda x: proc_token(x), words))
                }
                fout.write(json.dumps(data) + '\n')
                fout.flush()
            # for item in data['text']:
            #     dictionary.add_word(item)
            # qdar.set_postfix(dictSize=str(len(dictionary)))
        with open(args.dict+str(id), 'w') as fout:  # save dictionary for fast next process
            fout.write(json.dumps(dictionary.idx2word) + '\n')
    
    pool = []
    with open(args.input,encoding='utf-8') as csvfile:
        readCSV = csv.DictReader(csvfile, delimiter=',')
        for item in readCSV:
            pool.append(item)

    random.shuffle(pool)


    numIters = int(len(pool)/numProc)
    procList = []
    for pid in range(numProc):
        p = Process(target=sub_process, args=(numIters,pool[int(pid*numIters):int((pid+1)*numIters)],pid))
        p.start()
        procList.append(p)
    
    for pid in range(numProc):
        procList[pid].join()

        
        

    
