import gensim
import fnmatch
import os
import pickle
import numpy as np
from util import Dictionary
import json
# from symspellpy.symspellpy import SymSpell, Verbosity  # import the module

# initial_capacity = 83000
# # maximum edit distance per dictionary precalculation
# max_edit_distance_dictionary = 2
# prefix_length = 7
# sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,
#                      prefix_length)
# # load dictionary
# dictionary_path = os.path.join(os.path.dirname(__file__),
#                                "frequency_dictionary_en_82_765.txt")
# term_index = 0  # column of the term in the dictionary text file
# count_index = 1  # column of the term frequency in the dictionary text file
# if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
# 	print("Dictionary file not found")

# max_edit_distance_lookup = 2

model = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)



wordlist = []

# for dataset in ['yelp/']:
# 	filelist = os.listdir('../../Data/'+dataset)
# 	for file in filelist:
# 		with open('../../Data/'+dataset+file,'r') as f:
# 			line = f.readline()
# 			while line:
# 				# suggestions = sym_spell.lookup_compound(line, max_edit_distance_lookup)
# 				wordlist += line.split(' ')
# 				line = f.readline()

dictionary = Dictionary(path='./dict')
wordlist += dictionary.idx2word

wordlist.append('<unk>')
wordlist.append('<m_end>')
wordlist.append('@@START@@')
wordlist.append('@@END@@')
vocabs = set(wordlist)

print(len(vocabs))

wordDict = {}
word2vec = []
cnt=0
wastewords = []
for word in vocabs:
	if word in model.wv:
		word2vec.append(model.wv[word])
		wordDict[word] = cnt
		cnt += 1
	else:
		wastewords.append(word)
		word2vec.append(np.random.uniform(-1,1,300))
		wordDict[word] = cnt
		cnt += 1

print(len(wastewords))

word2vec = np.array(word2vec)
# with open('./word2vec', "wb") as fp:   #Pickling
np.save('word2vec.npy',word2vec)
# with open('./wordDict', "wb") as fp:   #Pickling
# 	pickle.dump(wordDict, fp)
with open('./Data/wordDict', 'w') as fout:  # save dictionary for fast next process
    fout.write(json.dumps(list(vocabs)) + '\n')
    fout.close()

# with open('./word2vec', "rb") as fp:   #Pickling
# 	word2vec = pickle.load(fp)
# with open('./wordDict', "rb") as fp:   #Pickling
# 	wordDict = pickle.load(fp)

# pass