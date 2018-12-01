from __future__ import print_function
from models import *

from util import Dictionary, get_args

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
import random
import os

import tqdm

from AttVisual import createHTML


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(data, volatile=False):
    """Package data for training / evaluation."""
    data = list(map(lambda x: json.loads(x), data))
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = Variable(torch.LongTensor(dat), requires_grad=False)
    targets = Variable(torch.LongTensor(targets), requires_grad=False)
    return dat.t(), targets


def evaluate():
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
            data, targets = package(data_val[i:min(len(data_val), i+args.batch_size)], volatile=True)
            if args.cuda:
                data = data.cuda()
                targets = targets.cuda()
            hidden = model.init_hidden(data.size(1))
            output, attention = model.forward(data, hidden)
            text = [dictionary.idx2word[data[i][0]] for i in range(len(data))]
            # visualization
            createHTML(text,attention.squeeze(),'test.html')


            output_flat = output.view(data.size(1), -1)
            # total_loss += criterion(output_flat, targets).data
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float())
    return total_correct.data.item() / len(data_val)


# def train(epoch_number, logger):
#     global best_val_loss, best_acc
#     model.train()
#     total_loss = 0
#     total_pure_loss = 0  # without the penalization term
#     # custom lr
#     if epoch_number>0 and epoch_number%8==0:
#         for param_group in optimizer.param_groups:
#                 param_group['lr'] = param_group['lr'] * 0.5

#     start_time = time.time()

#     numIters = int(len(data_train) / args.batch_size)
#     qdar = tqdm.tqdm(range(numIters),total= numIters,ascii=True)
#     for batch in qdar:
#         i = args.batch_size*batch
#     # for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
    
#         data, targets = package(data_train[i:i+args.batch_size], volatile=False)
#         if args.cuda:
#             data = data.cuda()
#             targets = targets.cuda()
#         hidden = model.init_hidden(data.size(1))
#         output, attention = model.forward(data, hidden)
#         loss = criterion(output.view(data.size(1), -1), targets)
#         total_pure_loss += loss.data
#         cur_p_ls = loss.data.item()

#         if attention is not None:  # add penalization term
#             attentionT = torch.transpose(attention, 1, 2).contiguous()
#             extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
#             loss += args.penalization_coeff * extra_loss
#         optimizer.zero_grad()
#         loss.backward()

#         nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#         optimizer.step()

#         total_loss += loss.data
#         cur_ls = loss.data.item()

#         qdar.set_postfix(loss= '{:5.4f}'.format(cur_ls), pure_loss='{:5.4f}'.format(cur_p_ls))

#         if batch % args.log_interval == 0 and batch > 0:
#             elapsed = time.time() - start_time
#             to_write = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f} \n'.format(
#                   epoch_number, batch, len(data_train) // args.batch_size,
#                   elapsed * 1000 / args.log_interval, total_loss.item() / args.log_interval,
#                   total_pure_loss.item() / args.log_interval)
#             # print()
#             logger.write(to_write)
#             logger.flush()
            
#             total_loss = 0
#             total_pure_loss = 0
#             start_time = time.time()

# #            for item in model.parameters():
# #                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
# #            print model.encoder.ws2.weight.grad.data
# #            exit()
#     evaluate_start_time = time.time()
#     val_loss, acc = evaluate()

#     print('-' * 89)
#     fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'.format((time.time() - evaluate_start_time), val_loss, acc)
#     print(fmt)
#     logger.write(fmt)
#     print('-' * 89)
#     # Save the model, if the validation loss is the best we've seen so far.
#     if not best_val_loss or val_loss < best_val_loss:
#         with open(args.save, 'wb') as f:
#             torch.save(model, f)
#         f.close()
#         best_val_loss = val_loss
#     # else:  # if loss doesn't go down, divide the learning rate by 5.
#     #     for param_group in optimizer.param_groups:
#     #         param_group['lr'] = param_group['lr'] * 0.2
#     if not best_acc or acc > best_acc:
#         with open(args.save[:-3]+'.best_acc.pt', 'wb') as f:
#             torch.save(model, f)
#         f.close()
#         best_acc = acc
#     # with open(args.save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
#     #     torch.save(model, f)
#     # f.close()


if __name__ == '__main__':
    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
    # assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)

    # best_val_loss = None
    # best_acc = None

    n_token = len(dictionary)
    model = Classifier({
        'dropout': args.dropout,
        'ntoken': n_token,
        'nlayers': args.nlayers,
        'nhid': args.nhid,
        'ninp': args.emsize,
        'pooling': 'all',
        'attention-unit': args.attention_unit,
        'attention-hops': args.attention_hops,
        'nfc': args.nfc,
        'dictionary': dictionary,
        'word-vector': args.word_vector,
        'class-number': args.class_number
    })
    checkpoint = torch.load('./exp/expsmall1/model-small.best_acc.pt', map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {}
    for k, v in checkpoint.state_dict().items():
        if(k in model_dict):
            pretrained_dict[k] = v
            # print(k)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if args.cuda:
        model = model.cuda()

    logger = open(args.log,'w')
    print(args)
    logger.write(str(args))
    I = Variable(torch.zeros(args.batch_size, args.attention_hops, args.attention_hops))
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda()

    print('Begin to load data.')
    # data_train = open(args.train_data).readlines()
    data_val = open(args.val_data).readlines()
    
    evaluate()
    logger.close()