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
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
# from pytorch_transformers import DistilBertForMaskedLM, DistilBertConfig

from distillation.distiller_from_finetuned import Distiller
from distillation.utils import git_log, logger, init_gpu_params, set_seed, parse_str2bool
from distillation.dataset import Dataset

from utils_glue import processors, output_modes, convert_examples_to_features
from run_glue import load_and_cache_examples


def main():
    parser = argparse.ArgumentParser(description="Training")

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory (log, checkpoints, parameters, etc.)")
    # parser.add_argument("--data_file", type=str, required=True,
    #                     help="The binarized file (tokenized + tokens_to_ids) and grouped by sequence.")
    # parser.add_argument("--token_counts", type=str, required=True,
    #                     help="The token counts in the data_file for MLM.")
    parser.add_argument("--force", action='store_true',
                        help="Overwrite output_dir if it already exists.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

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
    # parser.add_argument("--from_pretrained_weights", default=None, type=str,
    #                     help="Load student initialization checkpoint.")
    # parser.add_argument("--from_pretrained_config", default=None, type=str,
    #                     help="Load student initialization architecture config.")
    parser.add_argument("--teacher_name", default="bert-base-uncased", type=str,
                        help="The teacher model.")

    parser.add_argument("--use_hard_labels", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Whether to use hard labels instead of teacher logits in distillation.")    
    parser.add_argument("--temperature", default=2., type=float,
                        help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=1.0, type=float,
                        help="Linear weight for the distillation loss. Must be >=0.")
    # parser.add_argument("--alpha_mlm", default=0.5, type=float,
    #                     help="Linear weight for the MLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float,
                        help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument("--alpha_cos", default=0.0, type=float,
                        help="Linear weight of the cosine embedding loss. Must be >=0.")
    # parser.add_argument("--mlm_mask_prop", default=0.15, type=float,
    #                     help="Proportion of tokens for which we need to make a prediction.")
    # parser.add_argument("--word_mask", default=0.8, type=float,
    #                     help="Proportion of tokens to mask out.")
    # parser.add_argument("--word_keep", default=0.1, type=float,
    #                     help="Proportion of tokens to keep.")
    # parser.add_argument("--word_rand", default=0.1, type=float,
    #                     help="Proportion of tokens to randomly replace.")
    # parser.add_argument("--mlm_smoothing", default=0.7, type=float,
    #                     help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).")
    # parser.add_argument("--restrict_ce_to_mask", action='store_true',
    #                     help="If true, compute the distilation loss only the [MLM] prediction distribution.")

    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size (for each process).")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # parser.add_argument("--tokens_per_batch", type=int, default=-1,
    #                     help="If specified, modify the batches so that they have approximately this number of tokens.")
    # parser.add_argument("--shuffle", action='store_false',
    #                     help="If true, shuffle the sequence order. Default is true.")
    parser.add_argument("--group_by_size", action='store_false',
                        help="If true, group sequences that have similar length into the same batch. Default is true.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=50,
                        help="Gradient accumulation for larger training batches.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_prop", default=0.05, type=float,
                        help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
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
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--checkpoint_interval", type=int, default=-1,
                        help="Every how many epochs is a checkpoint saved at the end of the epoch.")
    parser.add_argument("--no_cuda", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Avoid using CUDA when available")
    parser.add_argument("--toy_mode", action='store_true', help="Toy mode for development.")
    parser.add_argument("--rich_eval", action='store_true', help="Rich evaluation (more metrics + mistake reporting).")

    # parser.add_argument("--generate_logits", type=parse_str2bool, default=False, const=True, nargs='?',
    #                     help="Instead of distillation, just generate teacher's logits for given data.")
    parser.add_argument("--augmentation_data_file", default=None, type=str,
                        help="File with augmentation sentences to be scored. If not provided, only the training set of the GLUE task will be considered.")
    parser.add_argument("--augmentation_type", default=None, type=str,
                        help="Type of transfer set augmentation (None, gpt-2 or rule-based).")
    parser.add_argument("--embeddings_from_teacher", type=parse_str2bool, default=False, const=True, nargs='?',
                        help="Take embeddings from the fine-tuned teacher, dimensionality reduced to fit the student.")

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
        logger.info(f'Param: {args}')
        with open(os.path.join(args.output_dir, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)


    ## GLUE TASK ##
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.model_name_or_path = args.teacher_name
    args.max_seq_length = args.max_position_embeddings
    args.model_type = "bert"
    args.output_mode = output_modes[args.task_name]

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.local_rank = -1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        device = torch.device("cuda", args.local_rank)
    args.device = device

    def create_features(sentences, args, label_list=[0]):
        features = convert_examples_to_features(sentences, 
            label_list=label_list,
            max_seq_length=args.max_seq_length, 
            tokenizer=tokenizer, output_mode="classification",
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            show_examples=False
        )
        dataset = {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device),
            "attention_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long).to(args.device),
            "token_type_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(args.device),
            "labels": torch.tensor([f.label_id for f in features], dtype=torch.long).to(args.device)
        }
        return dataset

    def add_logits(teacher, tokenizer, features, args):
        logger.info("Creating soft labels using the teacher model...")
        dataset = TensorDataset(features["input_ids"], features["attention_mask"], features["token_type_ids"], features["labels"])
        B = 2048 # use 128 on 6GB GPUs, or 512*N_GPUS on 12GB GPUs
        dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
        all_logits = None
        for i, batch in enumerate(dataloader):
            if args.n_gpu > 0:
                batch = tuple(t.to(f'cuda:{args.local_rank}') for t in batch)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            with torch.no_grad():
                (_, logits,) = teacher(**inputs)
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat([all_logits, logits], dim=0)
            logger.info("{}/{}".format(i*B, len(dataloader)*B))
        return all_logits

    ## TOKENIZER ##
    tokenizer = BertTokenizer.from_pretrained(args.teacher_name, do_lower_case=args.do_lower_case)
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    logger.info(f'Special tokens {special_tok_ids}')
    args.special_tok_ids = special_tok_ids
    teacher = None

    ## CACHED DATASET ##
    cached_dataset_file = os.path.join(args.data_dir, 'cached_train{}_msl{}_{}'.format(
        "" if args.augmentation_type is None else ("_augmented-" + args.augmentation_type),
        str(args.max_seq_length), 
        "logits" if not args.use_hard_labels else "hard"))
    if os.path.exists(cached_dataset_file):
        logger.info("Loading dataset from cached file %s", cached_dataset_file)
        d = torch.load(cached_dataset_file, map_location=args.device)
        train_dataset = TensorDataset(d["input_ids"], d["attention_mask"], d["token_type_ids"], d["labels"], d["logits"])
    else:
        ## TEACHER ##
        teacher = BertForSequenceClassification.from_pretrained(args.teacher_name) # take outputs[1] for the logits
        if args.n_gpu > 0:
            teacher.to(f'cuda:{args.local_rank}')
        logger.info(f'Teacher loaded from {args.teacher_name}.')

        ## ORIGINAL DATASET ##
        cached_original_set_file = os.path.join(args.data_dir, 'cached_train_msl{}_logits'.format(str(args.max_seq_length)))
        if os.path.exists(cached_original_set_file):
            cached_original_set = torch.load(cached_original_set_file, map_location=args.device)
            logger.info("Loading original dataset from cached file {}".format(cached_original_set_file))
        else:
            processor = processors[args.task_name]()
            original_sentences = processor.get_train_examples(args.data_dir)
            original_features = create_features(original_sentences, args, label_list=processor.get_labels())
            original_logits = add_logits(teacher, tokenizer, original_features, args)
            original_features["logits"] = original_logits
            cached_original_set = original_features
            logger.info("Saving original dataset with logits into cached file {}".format(cached_original_set_file))
            torch.save(cached_original_set, cached_original_set_file)

        ## AUGMENTATION DATASET ##
        if args.augmentation_type is not None:
            cached_augmentation_set_file = os.path.join(args.data_dir, 'cached_augmentation-{}_msl{}_logits'.format(
                args.augmentation_type,
                str(args.max_seq_length)))
            if os.path.exists(cached_augmentation_set_file):
                logger.info("Loading augmentation dataset from cached file {}".format(cached_augmentation_set_file))
                cached_augmentation_set = torch.load(cached_augmentation_set_file, map_location=args.device)
            else:
                processor = processors["sampled_{}".format(args.augmentation_type)]()
                augmentation_sentences = processor.get_examples(args.augmentation_data_file)
                augmentation_features = create_features(augmentation_sentences, args)
                augmentation_logits = add_logits(nn.DataParallel(teacher).cuda(), tokenizer, augmentation_features, args)
                augmentation_features["logits"] = augmentation_logits
                cached_augmentation_set = augmentation_features
                logger.info("Saving augmentation dataset with logits into cached file {}".format(cached_augmentation_set_file))
                torch.save(cached_augmentation_set, cached_augmentation_set_file)

        print(args.device)
        augmented_dataset = {name: torch.cat([cached_original_set[name], cached_augmentation_set[name]]) \
                             for name in ["input_ids", "attention_mask", "token_type_ids", "labels", "logits"]}
        logger.info("Saving augmented original dataset with logits into cached file {}".format(cached_dataset_file))
        torch.save(augmented_dataset, cached_dataset_file)
        train_dataset = TensorDataset(*[augmented_dataset[name] for name in ["input_ids", "attention_mask", "token_type_ids", "labels", "logits"]])

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    logger.info(f'Data loader created.')

    ## STUDENT ##
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
        initializer_range=0.02)
    
    if args.from_pretrained != "none":
        logger.info("Loading pre-trained student from: {}".format(args.from_pretrained))
        student = BertForSequenceClassification.from_pretrained(args.from_pretrained, config=student_config)
    else:
        student = BertForSequenceClassification(student_config)

        if args.embeddings_from_teacher:
            embeddings_file = os.path.join(args.teacher_name, "embeddings_h{}.pt".format(args.dim))
            if not os.path.exists(embeddings_file):
                teacher_state_dict = torch.load(os.path.join(args.teacher_name, "pytorch_model.bin"), map_location=args.device)
                embedding_weights = teacher_state_dict["bert.embeddings.word_embeddings.weight"]
                w_mean = torch.mean(embedding_weights, 0)
                embedding_weights = embedding_weights - w_mean.expand_as(embedding_weights)
                U, S, V = torch.svd(torch.t(embedding_weights))
                embeddings_reduced = torch.mm(embedding_weights, U[:,:args.dim])
                embeddings_state_dict = {"bert.embeddings.word_embeddings.weight": embeddings_reduced}
                torch.save(embeddings_state_dict, embeddings_file)
            else:
                embeddings_state_dict = torch.load(embeddings_file)

            param_keys = student.load_state_dict(embeddings_state_dict, strict=False)
            loaded_params = [p for p in student.state_dict() if p not in param_keys[0]]
            logger.info("Loaded the embedding weights: {}".format(loaded_params))

    # if args.from_pretrained_weights is not None:
    #     assert os.path.isfile(args.from_pretrained_weights)
    #     assert os.path.isfile(args.from_pretrained_config)
    #     logger.info(f'Loading pretrained weights from {args.from_pretrained_weights}')
    #     logger.info(f'Loading pretrained config from {args.from_pretrained_config}')
    #     stu_architecture_config = DistilBertConfig.from_json_file(args.from_pretrained_config)
    #     stu_architecture_config.output_hidden_states = True
    #     student = DistilBertForMaskedLM.from_pretrained(args.from_pretrained_weights,
    #                                                     config=stu_architecture_config)
    # else:
    #     args.vocab_size_or_config_json_file = args.vocab_size
    #     stu_architecture_config = DistilBertConfig(**vars(args), output_hidden_states=True)
    #     student = DistilBertForMaskedLM(stu_architecture_config)
    if args.n_gpu > 0:
        student.to(f'cuda:{args.local_rank}')
    logger.info(f'Student loaded.')

    ## DISTILLER ##
    torch.cuda.empty_cache()
    distiller = Distiller(params=args,
                          dataloader=train_dataloader,
                          student=student,
                          tokenizer=tokenizer)
    distiller.train()
    logger.info("Let's go get some drinks.")


if __name__ == "__main__":
    main()
