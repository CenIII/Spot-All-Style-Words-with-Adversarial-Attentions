from util import Dictionary
import json

print('Begin to load the dictionary.')
dictList = []
for i in range(16):
	dictList.append(Dictionary(path='./Data/data_clean/dict'+str(i)))

for i in range(15):
	for word in list(dictList[i+1].word2idx.keys()):
		dictList[0].add_word(word)

print('dict size: '+str(len(dictList[0])))

with open('./Data/data_clean/dictall', 'w') as fout:  # save dictionary for fast next process
    fout.write(json.dumps(dictList[0].idx2word) + '\n')

with open('./Data/data_clean/trainsetall', "w") as fout:
    for i in range(16):
        with open('./Data/data_clean/trainset'+str(i), "r") as infile:
            lines = infile.readlines()
            for line in lines:
            	fout.write(line)

