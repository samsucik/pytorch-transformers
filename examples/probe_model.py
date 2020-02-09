import senteval
import glob
import numpy as np
import argparse, os
import logging
from types import SimpleNamespace
import torch

from examples.distillation.utils import parse_str2bool
from examples.distillation.embedding_model import EmbeddingModel
from examples.distil_from_finetuned import create_word_level_tokenizer, numericalise_sentence
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertForSequenceClassification
from pytorch_transformers.configuration_bert import BertConfig

def dict_to_namespace(d):
    return SimpleNamespace(**d)

def num_labels(task):
    if task in ["CoLA", "SST-2"]: return 2
    else: return 57

def numericalise_batch(args, samples, tokenizer, cls_token, sep_token, pad_token_id):
    numericalised_samples = {"sentence": []}
    if args["model_type"] in ["BERT", "pretrained", "embedding_wordpiece", "embedding_word", "LSTM"]: numericalised_samples["attention_mask"] = []
    for example in samples:
        example_numericalised = numericalise_sentence(dict_to_namespace(args), " ".join(example), tokenizer,
                                                      cls_token=cls_token, sep_token=sep_token, 
                                                      pad_token_id=pad_token_id)
        if args["model_type"] in ["BERT", "pretrained", "embedding_wordpiece", "embedding_word", "LSTM"]:
            mask = (np.array(example_numericalised) != pad_token_id).astype(int)
            numericalised_samples["attention_mask"].append(torch.LongTensor(mask))
        numericalised_samples["sentence"].append(torch.LongTensor(example_numericalised))
    numericalised_samples["sentence"] = torch.nn.utils.rnn.pad_sequence(numericalised_samples["sentence"], batch_first=True, padding_value=args["pad_token_id"])
    if "attention_mask" in numericalised_samples:
        numericalised_samples["attention_mask"] = torch.nn.utils.rnn.pad_sequence(numericalised_samples["attention_mask"], batch_first=True, padding_value=0)
    return numericalised_samples

def embed(args, batch):
    with torch.no_grad():
        if args["model_type"] in ["BERT", "pretrained"]:
            # outputs = (logits, (hidden_states)); len(hidden_states) = L+1; 0th element is embedding layer
            # each element of hidden states is (B, MSL, H)
            _, hidden_states = args["model"](input_ids=batch["sentence"].to(args["device"]), 
                                                  attention_mask=batch["attention_mask"].to(args["device"]))
            if args["layer_to_probe"] == 47:
                layer = hidden_states[0] # take embedding layer
            elif args["layer_to_probe"] >= len(hidden_states) - 1 or args["layer_to_probe"] < 0:
                layer = hidden_states[-1] # take last layer
            else:
                layer = hidden_states[args["layer_to_probe"]+1] # layers are numbered from 1
            layer = layer.cpu()

            if args["embed_strategy"] == "avg":
                # average over the sequence length dimension
                seq_lens = torch.sum(batch["attention_mask"], dim=1)
                return [torch.mean(layer[i, :seq_len, :], dim=0).numpy() for i, seq_len in enumerate(seq_lens)]
            elif args["embed_strategy"] == "max":
                # take maximum over the sequence length dimension
                seq_lens = torch.sum(batch["attention_mask"], dim=1)
                return [torch.max(layer[i, :seq_len, :], dim=0)[0].numpy() for i, seq_len in enumerate(seq_lens)]
            elif args["embed_strategy"] == "single":    
                return layer[:, 0, :] # take hidden state at 0th position (the [CLS] token)
        elif args["model_type"] in ["embedding_word", "embedding_wordpiece"]:
            # (B, MSL, H)
            embeddings = args["model"](input_ids=batch["sentence"].to(args["device"])).cpu()
            if args["embed_strategy"] == "avg":
                # average over the sequence length dimension
                seq_lens = torch.sum(batch["attention_mask"], dim=1)
                return [torch.mean(embeddings[i, :seq_len, :], dim=0).numpy() for i, seq_len in enumerate(seq_lens)]
            elif args["embed_strategy"] == "max":
                # take maximum over the sequence length dimension
                seq_lens = torch.sum(batch["attention_mask"], dim=1)
                return [torch.max(embeddings[i, :seq_len, :], dim=0)[0].numpy() for i, seq_len in enumerate(seq_lens)]
            elif args["embed_strategy"] == "single":    
                return embeddings[:, 0, :] # take 0th embedding
        else: # LSTM student
            lengths = batch["attention_mask"].sum(axis=1)
            embeddings = args["model"]((batch["sentence"].to(args["device"]), lengths.to(args["device"]))).cpu() # (B, MSL, DxH)
            if args["embed_strategy"] == "avg":
                # average over the sequence length dimension
                seq_lens = torch.sum(batch["attention_mask"], dim=1)
                return [torch.mean(embeddings[i, :seq_len, :], dim=0).numpy() for i, seq_len in enumerate(seq_lens)]
            elif args["embed_strategy"] == "max":
                # take maximum over the sequence length dimension
                seq_lens = torch.sum(batch["attention_mask"], dim=1)
                return [torch.max(embeddings[i, :seq_len, :], dim=0)[0].numpy() for i, seq_len in enumerate(seq_lens)]
            elif args["embed_strategy"] == "single":    
                return embeddings[:, 0, :] # take 0th embedding

def prepare(args, samples):
    return

def batcher(args, batch):
    if args["pretrained"] is not None and "pretrained-config" not in args:
        args["pretrained-config"] = BertConfig.from_pretrained(args["pretrained"], num_labels=num_labels(args["glue_task"]))
        args["max_seq_length"] = args["pretrained-config"].max_position_embeddings

    if "tokenizer" not in args:
        logging.info("Creating a tokenizer...")
        if args["pretrained"] is not None:
            tokenizer = BertTokenizer.from_pretrained(args["pretrained"], do_lower_case=True)
        elif args["model_type"] in ["embedding_wordpiece", "embedding_word"]:
            args["max_seq_length"] = 128
            args["do_lower_case"] = True
            args["student_type"] = "LSTM" # so that no max_seq_len is used in the tokenizer and it doesn't use BERT-specific tokens (SEP, CLS)

            if args["use_word_vectors"]:
                vocab_file = os.path.join(args["glue_data_dir"], "transfer_set_vocab.txt")
                args["processed_word_vectors_file"] = os.path.join(args["glue_data_dir"], "word_vectors")
                tokenizer = create_word_level_tokenizer(dict_to_namespace(args), vocab_file, args["processed_word_vectors_file"], 
                                                        transfer_dataset_raw=None, transfer_dataset_raw_file=None)
            else:
                tokenizer = BertTokenizer.from_pretrained(args["model_dir"], do_lower_case=args["do_lower_case"], sep_token=None, cls_token=None)
        else: # tokenizer for student models
            model_args = torch.load(os.path.join(args["model_dir"], "training_args.bin"), map_location=args["device"])
            args["max_seq_length"] = model_args.max_seq_length
            if args["use_word_vectors"]:
                vocab_file = os.path.join(args["glue_data_dir"], "transfer_set_vocab.txt")
                args["processed_word_vectors_file"] = os.path.join(args["glue_data_dir"], "word_vectors")
                args["do_lower_case"] = model_args.do_lower_case
                tokenizer = create_word_level_tokenizer(dict_to_namespace(args), vocab_file, args["processed_word_vectors_file"], 
                                                        transfer_dataset_raw=None, transfer_dataset_raw_file=None)
            else:
                if args["model_type"] =="BERT":
                    tokenizer = BertTokenizer.from_pretrained(args["model_dir"], do_lower_case=model_args.do_lower_case)
                else: # LSTM has no sep_token or cls_token
                    tokenizer = BertTokenizer.from_pretrained(args["model_dir"], do_lower_case=model_args.do_lower_case, sep_token=None, cls_token=None)
        args["tokenizer"] = tokenizer
        
        args["cls_token"], args["sep_token"], pad_token = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token
        args["pad_token_id"] = tokenizer.convert_tokens_to_ids([pad_token])[0]

    if "model" not in args:
        logging.info("Loading the trained model...")
        if args["pretrained"] is not None:
            model = BertForSequenceClassification.from_pretrained(args["pretrained"], config=args["pretrained-config"])
            model.bert.encoder.output_hidden_states = True
        elif args["is_student"]:
            file =  glob.glob(os.path.join(args["model_dir"], "pytorch_model_best*.pt"))
            model = torch.load(file[0], map_location=args["device"])
            if args["student_type"] == "BERT": 
                model.bert.encoder.output_hidden_states = True
            else: # LSTM student
                model.output_hidden_states = True
                model.padding_idx = args["pad_token_id"]
                if args["layer_to_probe"] not in [-1, 0]:
                    raise ValueError("You can only probe the last layer (specified as -1 or 0) when using LSTM model.")
        elif args["model_type"] in ["embedding_word", "embedding_wordpiece"]:
            if args["model_type"] == "embedding_wordpiece":
                args["embedding_dimensionality"] = 1024
                args["embedding_type"] = "wordpiece"
                args["wordpiece_embeddings_file"] = os.path.join(args["model_dir"], "wordpiece_embeddings_teacher_h1024.pt")
                args["padding_idx"] = 0
            else:
                args["embedding_dimensionality"] = 300
                args["embedding_type"] = "word"
                args["processed_word_vectors_file"] = os.path.join(args["glue_data_dir"], "word_vectors")
                args["padding_idx"] = 1
            model = EmbeddingModel(args)
        else: # teacher model
            model = BertForSequenceClassification.from_pretrained(args["model_dir"])
            model.bert.encoder.output_hidden_states = True
        model.to(args["device"])
        model.eval()
        args["model"] = model

    batch_numericalised = numericalise_batch(args, batch, args["tokenizer"], args["cls_token"], args["sep_token"], args["pad_token_id"])
    torch.cuda.empty_cache()
    batch_embedded = embed(args, batch_numericalised)

    return batch_embedded

def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where the results will be stored.")
    parser.add_argument("--senteval_path", default="/home/sam/edi/minfp2/SentEval", type=str, required=False,
                        help="SentEval path.")
    parser.add_argument("--glue_task", default="CoLA", type=str, required=False,
                        help="One of: CoLA, SST-2, Sara.")
    parser.add_argument("--model_type", default="BERT", type=str, required=False,
                        help="One of: BERT, LSTM, pretrained, embedding_word, embedding_wordpiece.")
    parser.add_argument("--model_dir", default="teacher-CoLA", type=str, required=False,
                        help="Directory where trained model is saved.")
    parser.add_argument("--is_student", type=parse_str2bool, default=False,
                        help="Whether the model to probe is a student model or a teacher model.")
    parser.add_argument("--use_word_vectors", type=parse_str2bool, default=False,
                        help="Whether to use word vectors or not (only with student/embedding models).")
    parser.add_argument("--glue_data_dir", default="/home/sam/edi/minfp2/data/glue_data/CoLA", type=str, required=False,
                        help="Directory where the GLUE dataset is stored.")
    parser.add_argument("--layer_to_probe", default=-1, type=int, required=False,
                        help="Layer to probe; -1 equals to last layer. Numbering starts from 0.")
    parser.add_argument("--embed_strategy", default="single", type=str, required=False,
                        help="Strategy to use when extracting embeddings from a layer. Options: "
                        "'single' -- take the first (BERT) or last (LSTM) hidden state, "
                        "'avg' -- average over the (non-padding) hidden states, "
                        "'max' -- take maximum along each hidden dimension across all non-padding hidden states.")
    args = parser.parse_args()

    args.n_gpu = 1 if torch.cuda.is_available() else 0
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_type == "embedding_word":
        args.use_word_vectors = True
    elif args.model_type in ["embedding_wordpiece", "pretrained"]:
        args.use_word_vectors = False

    args.student_type = args.model_type
    args.pretrained = "bert-large-uncased" if args.model_type == "pretrained" else None
    args.cached_embeddings_file = os.path.join(args.glue_data_dir, 
        "probing_{}_{}_L{}_{}".format(args.pretrained if args.pretrained is not None else ("student" if args.is_student else "teacher"), 
            args.model_type, args.layer_to_probe, args.embed_strategy))
    args.dev_mode = not torch.cuda.is_available()
    
    logging.info(args)
   
    params = {'task_path': os.path.join(args.senteval_path, "data"), 
              'seed': 42,
              'usepytorch': True,
              'kfold': 10,
              'batch_size': 100 if not args.dev_mode else 15,
              **args.__dict__}
    params['classifier'] = {'nhid': 100,  # in paper they chose from [50, 100, 200]
                            'optim': 'adam', 
                            'batch_size': 64 if not args.dev_mode else 8,
                            'tenacity': 5, 
                            'epoch_size': 4,
                            'dropout': 0.1, # in paper they chose from [0.0, 0.1, 0.2],
                            'cudaEfficient': True,
                            }
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
    'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    nhids = [50, 100, 200]
    dropouts = [0.0, 0.1, 0.2]
    with open(os.path.join(args.out_dir, "results.csv"), "a") as f:
        f.write("task,devacc,acc,ndev,ntest,nhid,dropout\n")
    for task_name in transfer_tasks:
        for nhid in nhids:
            for dropout in dropouts:
                params["classifier"]["nhid"] = nhid
                params["classifier"]["dropout"] = dropout
                se = senteval.engine.SE(params, batcher, prepare)
                results = se.eval([task_name], dev_mode=args.dev_mode)
    
                with open(os.path.join(args.out_dir, "results.csv"), "a") as f:
                    for task, task_results in results.items():
                        line = "{},{},{},{},{},{},{}".format(task, task_results["devacc"], task_results["acc"], 
                                                             task_results["ndev"], task_results["ntest"], nhid, dropout)
                        f.write("{}\n".format(line))
                        logging.info(line)

if __name__ == '__main__':
    main()
