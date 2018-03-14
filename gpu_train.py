import argparse
import json
import numpy as np
from termcolor import cprint, colored as c
from tqdm import tqdm

import utils, data, metric, model


def get_labels(labels_file):
    
    with open(labels_file) as f:
        labels = json.load(f)

    return labels





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("files_path")
    parser.add_argument("--input_labels", default="input_chars.json")
    parser.add_argument("--output_labels", default="output_chars.json")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--num_epochs", default=48, type=int)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false", default=True)
    parser.add_argument("--model_name", default="GRU_Model", help="Model ID for saving files")
    args = parser.parse_args()

    input_chars = get_labels(args.input_labels)
    output_chars = ["<nop>", "<cap>"] + get_labels(args.output_labels)

    input_char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
    output_char2vec = utils.Char2Vec(chars=output_chars)
    input_size = input_char2vec.size
    output_size = output_char2vec.size
    hidden_size = input_size

    cprint("input_size is: " + c(input_size, 'green') + "; ouput_size is: " + c(output_size, 'green'))

    rnn = model.GruRNN(input_size, hidden_size, output_size, batch_size=args.batch_size, layers=args.num_layers, bi=args.bidirectional, cuda=args.cuda)
    if args.cuda:
        rnn.cuda()

    text_model = model.Model(rnn, input_char2vec, output_char2vec, cuda=args.cuda)
    text_model.setup_training(args.learning_rate)

    seq_length = 500

    for epoch_num in range(args.num_epochs):
        
        for batch_ind, (max_len, sources) in enumerate(tqdm(data.batch_gen(data.train_gen(args.files_path), args.batch_size))):
            
            # prepare the input and output chunks
            input_srcs = []; punc_targs = []
            for chunk in sources:
                input_source, punctuation_target = data.extract_punc(chunk, text_model.char2vec.chars, text_model.output_char2vec.chars)
                input_srcs.append(input_source)
                punc_targs.append(punctuation_target)
            
            # at the begining of the file, reset hidden to zero
            text_model.init_hidden_(random=False)
            seq_len = data.fuzzy_chunk_len(max_len, seq_length)
            for input_, target_ in zip(zip(*[data.chunk_gen(seq_len, src) for src in input_srcs]), 
                                       zip(*[data.chunk_gen(seq_len, tar, ["<nop>"]) for tar in punc_targs])):
                
                try:
                    text_model.forward(input_, target_)
                    text_model.descent()
                        
                except KeyError:
                    print(source)
                    raise KeyError
            
            if batch_ind%25 == 24:
                ### removing because this is just for jupyter cleanliness
                ###if batch_ind%100 == 99:
                    ###clear_output(wait=True)
                
                print('Epoch {:d} Batch {}'.format(epoch_num + 1, batch_ind + 1))
                print("=================================")
                punctuation_output = text_model.output_chars()
                ### added .cpu()
                ###plot_progress(text_model.embeded[0,:400].data.cpu().numpy().T, 
                              ###text_model.output[0,:400].data.cpu().numpy().T, 
                              ###text_model.softmax[0,:400].data.cpu().numpy().T,
                              ###np.array(text_model.losses))

                metric.print_pc(utils.flatten(punctuation_output), utils.flatten(target_))
                print()
                
            if batch_ind%100 == 99:
                
                validate_target = data.apply_punc(input_[0], target_[0])
                result = data.apply_punc(input_[0], 
                                         punctuation_output[0] )
                print(validate_target)
                print(result)
                
        # print('Dev Set Performance {:d}'.format(epoch_num))
        text_model.save('./data/{}_epoch-{}_batch-{}.tar'.format(args.model_name, epoch_num + 1, batch_ind + 1))


