# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training DistilBERT.
"""
import os
import argparse
import pickle
import json
import shutil
from tqdm import tqdm
from types import SimpleNamespace
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vectors 

from pytorch_transformers.tokenization_bert import BasicTokenizer
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from distillation.distiller_from_finetuned import Distiller
from distillation.utils import logger, init_gpu_params, set_seed, parse_str2bool
from distillation.dataset import Dataset
from distillation.tokenization_word import WordTokenizer
from distillation.bi_rnn import BiRNNModel

from utils_glue import processors, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--student_type", default="BERT", type=str, required=False,
                        help="Type of the student model. One of [BERT, LSTM].")    

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory (log, checkpoints, parameters, etc.)")
    
    parser.add_argument("--use_word_vectors", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Use word embeddings instead of wordpiece embeddings.")
    parser.add_argument("--word_vectors_dir", default=None, type=str, required=False,
                        help="Directory where pretrained word embeddings (e.g. word2vec) are located.")
    parser.add_argument("--transfer_set_tsv", default=None, type=str, required=False,
                        help="Transfer set as a TSV file, including the sentence, teacher logits and other attributes.")
    parser.add_argument("--word_vectors_file", default=None, type=str, required=False,
                        help="File name within word_vectors_dir.")

    parser.add_argument("--force", action='store_true',
                        help="Overwrite output_dir if it already exists.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: [cola, sst-2, sara]")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--vocab_size", default=30522, type=int,
                        help="The vocabulary size.")
    parser.add_argument("--max_position_embeddings", default=512, type=int,
                        help="Maximum sequence length we can model (including [CLS] and [SEP]).")
    parser.add_argument("--sinusoidal_pos_embds", action='store_false',
                        help="If true, the position embeddings are simply fixed with sinusoidal embeddings.")
    parser.add_argument("--n_layers", default=6, type=int,
                        help="Number of Transformer blocks.")
    parser.add_argument("--n_heads", default=12, type=int,
                        help="Number of heads in the self-attention module.")
    parser.add_argument("--dim", default=768, type=int,
                        help="Dimension through the network. Must be divisible by n_heads")
    parser.add_argument("--hidden_dim", default=3072, type=int,
                        help="Intermediate dimension in the FFN.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout.")
    parser.add_argument("--attention_dropout", default=0.1, type=float,
                        help="Dropout in self-attention.")
    parser.add_argument("--activation", default='gelu', type=str,
                        help="Activation to use in self-attention")
    parser.add_argument("--tie_weights_", action='store_false',
                        help="If true, we tie the embeddings matrix with the projection over the vocabulary matrix. Default is true.")

    parser.add_argument("--from_pretrained", default="none", type=str,
                        help="Load pretrained student initialization checkpoint (the config params must agree with the checkpoint).")
    parser.add_argument("--teacher_name", default="bert-base-uncased", type=str,
                        help="The teacher model.")

    parser.add_argument("--use_hard_labels", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Whether to use hard labels instead of teacher logits in distillation.")    
    parser.add_argument("--temperature", default=2., type=float,
                        help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=1.0, type=float,
                        help="Linear weight for the distillation loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float,
                        help="Linear weight of the MSE loss. Must be >=0.")

    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size (for each process).")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--group_by_size", action='store_false',
                        help="If true, group sequences that have similar length into the same batch. Default is true.")

    parser.add_argument("--optimizer", default="adam", type=str,
                        help="Optimizer to use (one of ['adam', 'adadelta']).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=50,
                        help="Gradient accumulation for larger training batches.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_epochs", default=0.05, type=float,
                        help="Number of epochs for linear warmup.")
    parser.add_argument("--lr_decay", type=parse_str2bool, default=False,
                        help="Whether to linearly decay the learning rate down to 0 or not.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam/Adadelta.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float,
                        help="Random initialization range.")

    parser.add_argument("--n_gpu", type=int, default=1,
                        help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56,
                        help="Random seed")

    parser.add_argument("--log_interval", type=int, default=10,
                        help="Tensorboard logging interval.")
    parser.add_argument('--log_examples', action='store_false',
                        help="Show input examples on the command line during evaluation. Enabled by default.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--checkpoint_interval", type=int, default=-1,
                        help="Every how many epochs is a checkpoint saved at the end of the epoch.")
    parser.add_argument("--no_cuda", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Avoid using CUDA when available")
    parser.add_argument("--toy_mode", action='store_true', help="Toy mode for development.")
    parser.add_argument("--rich_eval", action='store_true', help="Rich evaluation (more metrics + mistake reporting).")

    parser.add_argument("--augmentation_data_file", default=None, type=str,
                        help="File with augmentation sentences to be scored. If not provided, only the training set of the GLUE task will be considered.")
    parser.add_argument("--augmentation_type", default=None, type=str,
                        help="Type of transfer set augmentation (None, gpt-2 or rule-based).")
    
    parser.add_argument("--token_embeddings_from_teacher", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Take embeddings from the fine-tuned teacher, dimensionality reduced to fit the student.")
    parser.add_argument("--token_type_embedding_dimensionality", type=int, default=None,
                        help="Dimensionality of trained token type embeddings to be used (taken from teacher model). \
                              Relevant if token_embeddings_from_teacher or use_word_vectors set to True).")
    parser.add_argument("--token_embedding_dimensionality", type=int, default=None,
                        help="Dimensionality of trained wordpiece/word embeddings to be used. Relevant \
                              if token_embeddings_from_teacher or use_word_vectors set to True).")

    # LSTM-specific arguments
    parser.add_argument("--fc_size", type=int, default=400,
                        help="Fully connected layer size in LSTM student.")
    parser.add_argument("--hidden_size", type=int, default=300,
                        help="LSTM layer size.")
    parser.add_argument("--mode", default="multichannel", type=str,
                        help="Embedding mode. One of [rand, static, non-static, multichannel].")

    args = parser.parse_args()
   
    ## ARGS ##
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        if os.path.exists(args.output_dir):
            if not args.force:
                raise ValueError(f'Serialization dir {args.output_dir} already exists, but you have not precised wheter to overwrite it'
                                   'Use `--force` if you want to overwrite it')
            else:
                shutil.rmtree(args.output_dir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f'Experiment will be dumped and logged in {args.output_dir}')

        ### SAVE PARAMS ###
        logger.info("Param: {}".format(args))
        with open(os.path.join(args.output_dir, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    ## GLUE TASK ##
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: {}".format(args.task_name))
    if args.student_type == "LSTM":
        args.max_position_embeddings = -1
    args.max_seq_length = args.max_position_embeddings

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.local_rank = -1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        args.device = torch.device("cuda", args.local_rank)

    def add_logits(args, teacher, token_ids, attention_masks):
        logger.info("Generating logits using the teacher model")
        dataset = TensorDataset(token_ids, attention_masks)
        B = 512 # use 128 on 6GB GPUs, or 512*N_GPUS on 12GB GPUs
        dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
        all_logits = None
        teacher.eval()
        for i, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids = batch[0]
            attention_mask = batch[1]
            with torch.no_grad():
                logits = teacher(input_ids=input_ids, attention_mask=attention_mask)[0]
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat([all_logits, logits], dim=0)
        return all_logits

    def compress_embeddings_raunak(args, embeddings_tensor: torch.Tensor, new_dim=256, D1=5, D3=0):
        from sklearn.decomposition import PCA
        embeddings = embeddings_tensor.numpy()
        old_dim = embeddings.shape[1]
        print(embeddings.shape, old_dim)
        
        # preprocessing
        pca1 =  PCA(n_components=old_dim)
        z = embeddings - np.mean(embeddings, axis=0)
        pca1.fit(z)
        U1 = pca1.components_
        z1 = []
        for i, x in enumerate(z):
            for u in U1[0:D1]:
                x = x - np.dot(u.T, x) * u 
            z1.append(x)
        z1 = np.asarray(z1)

        # PCA dimensionality reduction
        pca2 =  PCA(n_components=new_dim)
        z2 = z1 - np.mean(z1, axis=0)
        z2 = pca2.fit_transform(z2)
        pca3 = PCA(n_components=new_dim)
        z3 = z2 - np.mean(z2, axis=0)
        pca3.fit(z3)
        U3 = pca3.components_

        # postprocessing
        if D3 > 0:
            U3 = pca3.components_
            z4 = []
            for i, x in enumerate(z3):
                for u in U3[0:D3]:
                    x = x - np.dot(u.T, x) * u 
                z4.append(x)
            embeddings_new = np.asarray(z4)
        else:
            embeddings_new = z3

        return torch.from_numpy(embeddings_new).float().to(args.device)

    def basic_tokenize(text, **tokenizer_kwargs): return BasicTokenizer(**tokenizer_kwargs).tokenize(text)
    
    def uniform_unk_init(a=-0.25, b=0.25): return lambda tensor: tensor.uniform_(a, b)
    
    def has_header(task_name, portion="train"):
        if task_name == "cola":
            if portion in ["train", "dev"]:
                return False
            else:
                return True
        elif task_name == "sst-2":
            return True
        elif task_name == "sara":
            return False
        else:
            raise ValueError("Unrecognised task name: {}".format(args.task_name))

    def numericalise_sentence(args, sentence, tokenizer, cls_token="[CLS]", sep_token="[SEP]", pad_token_id=0, force_pad=0):
        tokens = tokenizer.tokenize(sentence)
        if args.max_seq_length > 0 and len(tokens) > args.max_seq_length - 2:
            tokens = tokens[:(args.max_seq_length - 2)]
        tokens = ([cls_token] if cls_token is not None else []) + tokens + ([sep_token] if cls_token is not None else [])
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if args.max_seq_length > 0 or force_pad > 0:
            token_ids += [pad_token_id] * (max(args.max_seq_length, force_pad) - len(token_ids))
        return token_ids

    def get_original_dev_dataset_fields(args):
        text_field = Field(use_vocab=False, tokenize=lambda s: s, batch_first=True, lower=args.do_lower_case)
        label_field = Field(sequential=False, use_vocab=False, batch_first=True)
        if args.task_name == "cola":
            # gj04    0   *   They drank the pub.
            return [("guid", None), ("label", label_field), ("acceptability", None), ("sentence", text_field)]
        elif args.task_name == "sst-2":
            # hide new secretions from the parental units     0
            return [("sentence", text_field), ("label", None)]
        elif args.task_name == "sara":
            # 0 Yep that's fine
            return [("label", None), ("sentence", text_field)]
        else:
            raise ValueError("Unrecognised task name: {}".format(args.task_name))

    def get_original_train_dataset_fields(args):
        text_field = Field(use_vocab=False, tokenize=lambda s: s, lower=args.do_lower_case, batch_first=True)
        if args.task_name == "cola":
            # gj04    0   *   They drank the pub.
            return [("guid", None), ("label", None), ("acceptability", None), ("sentence", text_field)]
        elif args.task_name == "sst-2":
            # hide new secretions from the parental units     0
            return [("sentence", text_field), ("label", None)]
        elif args.task_name == "sara":
            # 0 Yep that's fine
            return [("label", None), ("sentence", text_field)]
        else:
            raise ValueError("Unrecognised task name: {}".format(args.task_name))

    def get_n_classes(task_name):
        if task_name == "cola":
            n_classes = 2
        elif task_name == "sst-2":
            n_classes = 2
        elif task_name == "sara":
            n_classes = 57
        else:
            raise ValueError("Unrecognised task name: {}".format(task_name))
        return n_classes    

    def get_augmented_dataset_fields(args):
        n_classes = get_n_classes(args.task_name)
        text_field = Field(tokenize=basic_tokenize, lower=args.do_lower_case, batch_first=True)
        logit_fields = [("logit_{}".format(i), Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)) 
                        for i in range(n_classes)]

        # return [("label", None), ("sentence", text_field)] + logit_fields
        return [("sentence", text_field)] + logit_fields

    def create_raw_transfer_set(args, cached_dataset_file):
        if not os.path.exists(cached_dataset_file):
            logger.info("Creating the raw transfer set with teacher logits")
            # get original set
            fields = get_original_train_dataset_fields(args)
            original_train_file = os.path.join(args.data_dir, "train.tsv")
            original_train_set = TabularDataset(original_train_file, format="tsv", fields=fields, skip_header=has_header(args.task_name))

            # get augmentation set
            augmentation_file = os.path.join(args.data_dir, "sampled_sentences")
            augmentation_fields = [("sentence", Field(use_vocab=False, tokenize=lambda s: s[2:], lower=args.do_lower_case, batch_first=True))] # drop first 2 chars!
            augmentation_set = TabularDataset(augmentation_file, format="tsv", fields=augmentation_fields, skip_header=False)

            # merge to create entire transfer set
            combined_transfer_set = original_train_set
            combined_transfer_set.examples += augmentation_set.examples

            # TO-DO delete once everything works
            # combined_transfer_set.examples = combined_transfer_set.examples[:5]

            # numericalise and pad the sentences
            tokenizer_teacher = BertTokenizer.from_pretrained(args.teacher_name, do_lower_case=args.do_lower_case)
            numericalised_transfer_set = []
            attention_masks = []
            cls_token, sep_token, pad_token = tokenizer_teacher.cls_token, tokenizer_teacher.sep_token, tokenizer_teacher.pad_token
            pad_token_id = tokenizer_teacher.convert_tokens_to_ids([pad_token])[0]

            logger.info("Numericalising the transfer set using the teacher's tokenizer")
            for example in tqdm(combined_transfer_set.examples):
                example_numericalised = numericalise_sentence(args, example.sentence, tokenizer_teacher, 
                                                              cls_token=cls_token, sep_token=sep_token, 
                                                              pad_token_id=pad_token_id, force_pad=128)
                mask = (np.array(example_numericalised) != pad_token_id).astype(int)
                attention_masks.append(torch.LongTensor(mask))
                numericalised_transfer_set.append(torch.LongTensor(example_numericalised))
            numericalised_transfer_set = torch.stack(numericalised_transfer_set)
            attention_masks = torch.stack(attention_masks)
            
            # add teacher logits
            teacher = BertForSequenceClassification.from_pretrained(args.teacher_name) # take outputs[1] for the logits
            teacher.to(args.device)
            logits = add_logits(args, teacher, numericalised_transfer_set, attention_masks)

            # """
            # write sentences and logits into TSV
            with open(cached_dataset_file, "w") as writer:
                n_logits = logits.shape[1]
                writer.write("sentence\t" + "\t".join(["logit_{}".format(i) for i in range(n_logits)]) + "\n")
                for i, ex in enumerate(combined_transfer_set.examples):
                    soft_labels = logits[i, :].cpu().numpy()
                    soft_labels_str = "\t".join([str(logit) for logit in soft_labels])
                    writer.write("{}\t".format(ex.sentence) + soft_labels_str + "\n")
            """
            n_logits = logits.shape[1]
            for i, ex in enumerate(combined_transfer_set.examples):
                soft_labels = logits[i, :].cpu().numpy()
                soft_labels_str = "\t".join([str(logit) for logit in soft_labels])
                print("{}\t".format(ex.sentence) + soft_labels_str)
            exit(0)
            """
        
        # read the dataset from TSV
        logger.info("Loading raw transfer set from TSV file {}".format(cached_dataset_file))
        fields = get_augmented_dataset_fields(args)
        train_set = TabularDataset(cached_dataset_file, format="tsv", fields=fields, skip_header=True)

        return train_set

    def create_numerical_transfer_set(args, transfer_dataset_numerical_file, tokenizer, transfer_dataset_raw=None):
        if not os.path.exists(transfer_dataset_numerical_file):
            logger.info("Creating numerical transfer dataset from the raw one")
            transfer_dataset_numerical = {"sentence": [], "logits": []}
            if args.student_type == "BERT": transfer_dataset_numerical["attention_mask"] = []
            cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
            pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            n_classes = get_n_classes(args.task_name)
            logit_names = ["logit_{}".format(c) for c in range(n_classes)]
            logger.info("Numericalising the transfer set using the student's tokenizer")
            for example in tqdm(transfer_dataset_raw.examples):
                # here, the sentence is already tokenized into words, which is OK (when using wordpieces, not words)
                # ONLY IF BertTokenizer has do_basic_tokenize set to True -- but that's the (sensible) default.
                sentence = " ".join(example.sentence)
                example_numericalised = numericalise_sentence(args, sentence, tokenizer, 
                                                              cls_token=cls_token, sep_token=sep_token, 
                                                              pad_token_id=pad_token_id)
                example_logits = [float(getattr(example, logit_name)) for logit_name in logit_names]
                if args.student_type == "BERT":
                    mask = (np.array(example_numericalised) != pad_token_id).astype(int)
                    transfer_dataset_numerical["attention_mask"].append(torch.LongTensor(mask))
                transfer_dataset_numerical["sentence"].append(torch.LongTensor(example_numericalised))
                transfer_dataset_numerical["logits"].append(torch.FloatTensor(example_logits))
            torch.save(transfer_dataset_numerical, transfer_dataset_numerical_file)
        else:
            logger.info("Loading the numerical transfer dataset from binary file: {}".format(transfer_dataset_numerical_file))
            transfer_dataset_numerical = torch.load(transfer_dataset_numerical_file, map_location=args.device)
        
        # padding according to longest sentence when using LSTM
        if not args.max_seq_length > 0:
            lens = [len(sent) for sent in transfer_dataset_numerical["sentence"]]
            transfer_dataset_numerical["sentence_length"] = torch.LongTensor(lens)
            pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            transfer_dataset_numerical = pad_sentences_to_longest(args, transfer_dataset_numerical, max(lens), pad_token_id)
        else:
            transfer_dataset_numerical["sentence_length"] = torch.LongTensor([0 for i in range(len(transfer_dataset_numerical["sentence"]))])

        if args.student_type == "BERT":
            transfer_dataset_numerical["attention_mask"] = torch.stack(transfer_dataset_numerical["attention_mask"])
        transfer_dataset_numerical["sentence"] = torch.stack(transfer_dataset_numerical["sentence"])
        transfer_dataset_numerical["logits"] = torch.stack(transfer_dataset_numerical["logits"])
        if args.student_type == "BERT":
            transfer_dataset = TensorDataset(transfer_dataset_numerical["sentence"], transfer_dataset_numerical["logits"], 
                                             transfer_dataset_numerical["sentence_length"], transfer_dataset_numerical["attention_mask"])
        else:
            transfer_dataset = TensorDataset(transfer_dataset_numerical["sentence"], transfer_dataset_numerical["logits"], 
                                             transfer_dataset_numerical["sentence_length"])
        transfer_dataset_sampler = RandomSampler(transfer_dataset)
        transfer_dataset = DataLoader(transfer_dataset, sampler=transfer_dataset_sampler, batch_size=args.batch_size)

        return transfer_dataset

    def pad_sentences_to_longest(args, dataset, target_len, pad_token_id):
        for i in range(len(dataset["sentence"])):
            sent_len = len(dataset["sentence"][i])
            padding = torch.LongTensor([pad_token_id]*(target_len - sent_len))
            dataset["sentence"][i] = torch.cat((dataset["sentence"][i].to(args.device), padding.to(args.device))).to(args.device)
        return dataset
    
    def create_numerical_dev_set(args, dev_dataset_numerical_file, tokenizer):
        if not os.path.isfile(dev_dataset_numerical_file):
            fields = get_original_dev_dataset_fields(args)
            dev_dataset_raw_file = os.path.join(args.data_dir, "dev.tsv")
            dev_dataset_raw = TabularDataset(dev_dataset_raw_file, format="tsv", fields=fields, skip_header=has_header(args.task_name))
            cls_token, sep_token, pad_token = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token
            pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
            dev_dataset_numerical = {"sentence": [], "label": []}
            if args.student_type == "BERT": dev_dataset_numerical["attention_mask"] = []
            for example in tqdm(dev_dataset_raw.examples):
                # here, the sentence is already tokenized into words, which is OK (when using wordpieces, not words)
                # ONLY IF BertTokenizer has do_basic_tokenize set to True -- but that's the (sensible) default.
                example_numericalised = numericalise_sentence(args, example.sentence, tokenizer, 
                                                              cls_token=cls_token, sep_token=sep_token, 
                                                              pad_token_id=pad_token_id)
                if args.student_type == "BERT":
                    mask = (np.array(example_numericalised) != pad_token_id).astype(int)
                    dev_dataset_numerical["attention_mask"].append(torch.LongTensor(mask))
                dev_dataset_numerical["sentence"].append(torch.LongTensor(example_numericalised))
                dev_dataset_numerical["label"].append(torch.LongTensor([int(example.label)]))
            torch.save(dev_dataset_numerical, dev_dataset_numerical_file)
        else:
            dev_dataset_numerical = torch.load(dev_dataset_numerical_file, map_location=args.device)
        
        # padding according to longest sentence when using LSTM
        if args.student_type == "LSTM":
            lens = [len(sent) for sent in dev_dataset_numerical["sentence"]]
            dev_dataset_numerical["sentence_length"] = torch.LongTensor(lens)
            pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token_id])[0]
            dev_dataset_numerical = pad_sentences_to_longest(args, dev_dataset_numerical, max(lens), pad_token_id)
        else:
            dev_dataset_numerical["sentence_length"] = torch.LongTensor([0 for i in range(len(dev_dataset_numerical["sentence"]))])
            
        if args.student_type == "BERT": dev_dataset_numerical["attention_mask"] = torch.stack(dev_dataset_numerical["attention_mask"])
        dev_dataset_numerical["sentence"] = torch.stack(dev_dataset_numerical["sentence"])
        dev_dataset_numerical["label"] = torch.stack(dev_dataset_numerical["label"])
        if args.student_type == "BERT":
            dev_dataset = TensorDataset(dev_dataset_numerical["sentence"], dev_dataset_numerical["label"], 
                                        dev_dataset_numerical["sentence_length"], dev_dataset_numerical["attention_mask"])
        else:
            dev_dataset = TensorDataset(dev_dataset_numerical["sentence"], dev_dataset_numerical["label"], 
                                        dev_dataset_numerical["sentence_length"])
        dev_dataset_sampler = SequentialSampler(dev_dataset)
        dev_dataset = DataLoader(dev_dataset, sampler=dev_dataset_sampler, batch_size=args.per_gpu_eval_batch_size)
        
        return dev_dataset

    def evaluate_model(params: SimpleNamespace):
        # params contains:
        # - dataset
        # - model
        # - task_name
        # - device
        # - student type
        params.model.eval()
        preds, targets = None, None
        for batch in params.dataset:
            batch = tuple(t.to(params.device) for t in batch)
            with torch.no_grad():
                # dev dataset contains: [sentence, label, sentence_length, attention_mask]
                if params.student_type == "BERT":
                    logits = params.model(batch[0], attention_mask=batch[3])[0]
                # dev dataset contains: [sentence, label, sentence_length]
                else: # feed LSTM also with sentence lengths
                    logits = params.model((batch[0], batch[2]))
            labels_pred = logits.max(1)[1]
            if preds is None:
                preds = labels_pred.detach().cpu().numpy()
                targets = batch[1].detach().cpu().numpy()
            else:
                preds = np.append(preds, labels_pred.detach().cpu().numpy(), axis=0)
                targets = np.append(targets, batch[1].detach().cpu().numpy(), axis=0)
        result = compute_metrics(params.task_name, preds, targets)
        return result

    def create_word_level_tokenizer(args, vocab_file, processed_word_vectors_file, transfer_dataset_raw, transfer_dataset_raw_file):
        if not os.path.isfile(vocab_file) or not os.path.isfile(processed_word_vectors_file):
            if transfer_dataset_raw is None:
                logger.info("Creating the raw (TSV) transfer set as it doesn't exist but is needed to construct student's word-level tokenizer")
                transfer_dataset_raw = create_raw_transfer_set(args, transfer_dataset_raw_file)
            logger.info("Creating vocab file from the transfer set")
            vectors = Vectors(name=args.word_vectors_file, cache=args.word_vectors_dir, unk_init=uniform_unk_init())
            text_field = transfer_dataset_raw.fields["sentence"]
            text_field.build_vocab(transfer_dataset_raw, vectors=vectors, min_freq=1)
            with open(vocab_file, "w", encoding="utf-8") as writer:
                for word in text_field.vocab.itos:
                    writer.write(word + u'\n')
            torch.save(text_field.vocab.vectors, processed_word_vectors_file)

        # create tokenizer using that vocab file name and config from teacher directory
        if args.student_type == "BERT":
            special_tokens_map = {"unk_token": "<unk>", "pad_token": "<pad>", "cls_token": "<cls>", "sep_token": "<sep>"}
        else:
            special_tokens_map = {"unk_token": "<unk>", "pad_token": "<pad>"}

        tokenizer = WordTokenizer(vocab_file=vocab_file, do_lower_case=args.do_lower_case, 
                                  max_len=(args.max_seq_length if args.student_type == "BERT" else None))
        tokenizer.add_special_tokens(special_tokens_map)

        special_tok_ids = {}
        for tok_name, tok_symbol in special_tokens_map.items():
            idx = tokenizer.encode(tok_symbol)[0]
            special_tok_ids[tok_name] = idx
        logger.info("Special tokens: {}".format(special_tok_ids))
        args.special_tok_ids = special_tok_ids
        return tokenizer

    ## TRANSFER SET
    transfer_dataset_numerical_file = os.path.join(args.data_dir, "train{}_{}_msl{}_{}.bin".format(
            "" if args.augmentation_type is None else ("_augmented-" + args.augmentation_type), 
            "word" if args.use_word_vectors else "wordpiece", str(args.max_seq_length), args.student_type))
    transfer_dataset_raw_file = os.path.join(args.data_dir, "train{}_scored.tsv".format(
            "" if args.augmentation_type is None else ("_augmented-" + args.augmentation_type)))
    # transfer_dataset_raw_file = os.path.join(args.data_dir, "cached_train_augmented-gpt-2_msl128_logits_bilstm-toy.csv")
    # STAGE 1: create and store sentence and logits as TSV - model-agnostic raw transfer set.
    if not os.path.isfile(transfer_dataset_numerical_file):
        transfer_dataset_raw = create_raw_transfer_set(args, transfer_dataset_raw_file)
    else:
        transfer_dataset_raw = None

    ## STUDENT'S TOKENIZER
    if args.use_word_vectors:
        vocab_file = os.path.join(args.data_dir, "transfer_set_vocab.txt")
        args.processed_word_vectors_file = os.path.join(args.data_dir, "word_vectors")
        tokenizer = create_word_level_tokenizer(args, vocab_file, args.processed_word_vectors_file, 
                                                transfer_dataset_raw, transfer_dataset_raw_file)
        args.vocab_size = len(tokenizer)
    else:
        if args.student_type == "LSTM":
            tokenizer = BertTokenizer.from_pretrained(args.teacher_name, do_lower_case=args.do_lower_case,
                                                      sep_token=None, cls_token=None)
        else:
            tokenizer = BertTokenizer.from_pretrained(args.teacher_name, do_lower_case=args.do_lower_case)
        special_tok_ids = {}
        for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
            idx = tokenizer.all_special_tokens.index(tok_symbol)
            special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
        logger.info("Special tokens: {}".format(special_tok_ids))
        args.special_tok_ids = special_tok_ids

    # STAGE 2: create and store numericalised transfer set (input ids, logits) as binary - model-specific transfer set.
    transfer_dataset = create_numerical_transfer_set(args, transfer_dataset_numerical_file, tokenizer, transfer_dataset_raw)

    # find maximum sequence length present in transfer set
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    max_len_found = 0
    for sample in transfer_dataset:
        sents = sample[0].cpu().numpy()
        for s in sents:
            length = np.array(s != pad_token_id).astype(int).sum()
            max_len_found = max(max_len_found, length)
    logger.info("Longest transfer set example has length: {}".format(max_len_found))

    ## DEV SET
    if args.evaluate_during_training:
        dev_dataset_numerical_file = os.path.join(args.data_dir, "dev_{}_msl{}_{}.bin".format(
            "word" if args.use_word_vectors else "wordpiece", str(args.max_seq_length), args.student_type))
        dev_dataset = create_numerical_dev_set(args, dev_dataset_numerical_file, tokenizer)
        evaluate_fn = evaluate_model
    else:
        dev_dataset = None
        evaluate_fn = None

    ## STUDENT
    args.use_learned_embeddings = args.token_embeddings_from_teacher or args.use_word_vectors
    args.n_classes = get_n_classes(args.task_name)

    if args.student_type == "LSTM":
        torch.cuda.deterministic = True
        student = BiRNNModel(args)
    else:
        # specify embedding dimensionality only if it's different from the hidden size of the student
        student_config = BertConfig(
            vocab_size_or_config_json_file=args.vocab_size,
            hidden_size=args.dim,
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            intermediate_size=args.hidden_dim,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.attention_dropout,
            max_position_embeddings=args.max_position_embeddings,
            hidden_act=args.activation,
            initializer_range=args.initializer_range,
            token_embedding_dimensionality=args.token_embedding_dimensionality,
            token_type_embedding_dimensionality=args.token_type_embedding_dimensionality,
            embedding_mode=args.mode,
        )    
    # Either load full learned student model
    if args.from_pretrained != "none":
        logger.info("Loading pre-trained student from: {}".format(args.from_pretrained))
        if args.student_type == "LSTM":
            state_dict = torch.load(args.from_pretrained, map_location=args.device)
            param_keys = student.load_state_dict(state_dict, strict=False)
            loaded_params = [p for p in student.state_dict() if p not in param_keys[0]]
            num_loaded_params = sum([student.state_dict()[p].numel() for p in loaded_params])
            num_all_params = sum([p.numel() for n, p in student.state_dict().items()])
            print("Loaded {} parameters (out of total {}) into: {}".format(num_loaded_params, num_all_params, loaded_params))
        else:
            student = BertForSequenceClassification.from_pretrained(args.from_pretrained, config=student_config)
    # Or create student initialised from scratch, or with only embedding layer learned
    else:
        teacher_token_type_embedding_name = "bert.embeddings.token_type_embeddings.weight"
        teacher_token_embedding_name = "bert.embeddings.word_embeddings.weight"
        if args.student_type == "BERT":
            student = BertForSequenceClassification(student_config)
            if args.mode == "rand":
                token_embedding_names = []
            elif args.mode == "static":
                token_embedding_names = ["bert.embeddings.word_embeddings_static.weight"]
                token_type_embedding_names = ["bert.embeddings.token_type_embeddings_static.weight"]
            elif args.mode == "non-static":
                token_embedding_names = ["bert.embeddings.word_embeddings.weight"]
                token_type_embedding_names = ["bert.embeddings.token_type_embeddings.weight"]
            elif args.mode == "multichannel":
                token_embedding_names = ["bert.embeddings.word_embeddings.weight", "bert.embeddings.word_embeddings_static.weight"]
                token_type_embedding_names = ["bert.embeddings.token_type_embeddings.weight", "bert.embeddings.token_type_embeddings_static.weight"]
        else:
            if args.mode == "rand":
                token_embedding_names = []
            elif args.mode == "static":
                token_embedding_names = ["static_embed.weight"]
            elif args.mode == "non-static":
                token_embedding_names = ["non_static_embed.weight"]
            elif args.mode == "multichannel":
                token_embedding_names = ["non_static_embed.weight", "static_embed.weight"]

        if args.use_learned_embeddings:
            embeddings_to_load = {}

            # retrieve learned token type embeddings to be used with learned wordpiece/word embeddings
            if args.student_type == "BERT":
                logger.info("Retrieving learned token type embeddings from the teacher")
                token_type_embeddings_file = os.path.join(args.teacher_name, "token_type_embeddings_teacher_h{}.pt"\
                    .format(args.token_type_embedding_dimensionality))
                if not os.path.isfile(token_type_embeddings_file):
                    teacher_state_dict = torch.load(os.path.join(args.teacher_name, "pytorch_model.bin"), map_location=torch.device("cpu"))
                    embedding_weights = teacher_state_dict[teacher_token_type_embedding_name]
                    assert args.token_type_embedding_dimensionality == embedding_weights.shape[-1]
                    token_type_embedding_state_dict = {teacher_token_type_embedding_name: embedding_weights}
                    torch.save(token_type_embedding_state_dict, token_type_embeddings_file)
                else:
                    token_type_embedding_state_dict = torch.load(token_type_embeddings_file, map_location=args.device)
                for name in token_type_embedding_names:
                    embeddings_to_load[name] = token_type_embedding_state_dict[teacher_token_type_embedding_name]

            # retrieve learned wordpiece embeddings from teacher model
            if args.token_embeddings_from_teacher:
                logger.info("Initialising student wordpiece embedding parameters from the teacher")
                embeddings_file = os.path.join(args.teacher_name, "wordpiece_embeddings_teacher_h{}.pt".format(args.token_embedding_dimensionality))
                if not os.path.exists(embeddings_file):
                    teacher_state_dict = torch.load(os.path.join(args.teacher_name, "pytorch_model.bin"), map_location=torch.device("cpu"))
                    embedding_weights = teacher_state_dict[teacher_token_embedding_name]
                    token_embedding_state_dict = {teacher_token_embedding_name: embedding_weights}
                    assert args.token_embedding_dimensionality == embedding_weights.shape[-1]
                    torch.save(token_embedding_state_dict, embeddings_file)
                else:
                    token_embedding_state_dict = torch.load(embeddings_file, map_location=args.device)
                for token_embedding_name in token_embedding_names:
                    embeddings_to_load[token_embedding_name] = token_embedding_state_dict[teacher_token_embedding_name]

            # retrieve learned word embeddings, e.g. word2vec
            elif args.use_word_vectors:
                logger.info("Initialising student word embedding parameters from learned word vectors")
                # add to the word vectors randomly initialised embeddings for any additional special tokens
                word_embeddings = torch.load(args.processed_word_vectors_file, map_location=args.device)
                # add special token embeddings for BERT
                if args.student_type == "BERT":
                    word_embeddings = torch.cat((word_embeddings, 
                                                 torch.FloatTensor(len(tokenizer.added_tokens_encoder), word_embeddings.shape[1])\
                                                 .uniform_(-0.25, 0.25).to(args.device)))
                for token_embedding_name in token_embedding_names:
                    embeddings_to_load[token_embedding_name] = word_embeddings

            param_keys = student.load_state_dict(embeddings_to_load, strict=False)
            loaded_params = [p for p in student.state_dict() if p not in param_keys[0]]
            num_loaded_params = sum([student.state_dict()[p].numel() for p in loaded_params])
            num_all_params = sum([p.numel() for n, p in student.state_dict().items()])
            logger.info("Loaded {} parameters (out of total {}) into: {}".format(num_loaded_params, num_all_params, loaded_params))
    student.to(args.device)
    logger.info("Student model created.")
    
    ## DISTILLER
    torch.cuda.empty_cache()
    distiller = Distiller(params=args,
                          dataset_train=transfer_dataset,
                          dataset_eval=dev_dataset,
                          student=student,
                          evaluate_fn=evaluate_fn,
                          student_type=args.student_type)
    distiller.train()
    logger.info("Let's go get some drinks.")

if __name__ == "__main__":
    main()
