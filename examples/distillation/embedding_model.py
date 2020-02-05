import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel(nn.Module):
    def __init__(self, args):
        """
        padding_idx
        model_dir (wordpiece)
        processed_word_vectors_file (word)
        embedding_dimensionality
        embedding_type: "wordpiece" or "word"
        device
        """
        super(EmbeddingModel, self).__init__()
        # self.args = args

        # teacher_token_type_embedding_name = "bert.embeddings.token_type_embeddings.weight"
        teacher_token_embedding_name = "bert.embeddings.word_embeddings.weight"
        token_embedding_names = ["embeddings"]
        embeddings_to_load = {}

        # retrieve learned token type embeddings to be used with learned wordpiece/word embeddings
        # if args.student_type == "BERT":
        #     logger.info("Retrieving learned token type embeddings from the teacher")
        #     token_type_embeddings_file = os.path.join(args.teacher_name, "token_type_embeddings_teacher_h{}.pt"\
        #         .format(args.token_type_embedding_dimensionality))
        #     if not os.path.isfile(token_type_embeddings_file):
        #         teacher_state_dict = torch.load(os.path.join(args.teacher_name, "pytorch_model.bin"), map_location=torch.device("cpu"))
        #         embedding_weights = teacher_state_dict[teacher_token_type_embedding_name]
        #         assert args.token_type_embedding_dimensionality == embedding_weights.shape[-1]
        #         token_type_embedding_state_dict = {teacher_token_type_embedding_name: embedding_weights}
        #         torch.save(token_type_embedding_state_dict, token_type_embeddings_file)
        #     else:
        #         token_type_embedding_state_dict = torch.load(token_type_embeddings_file, map_location=args.device)
        #     for name in token_type_embedding_names:
        #         embeddings_to_load[name] = token_type_embedding_state_dict[teacher_token_type_embedding_name]

        # retrieve learned wordpiece embeddings from teacher model
        if args.embedding_type == "wordpiece":
            logger.info("Initialising wordpiece embedding parameters from the teacher")
            # embeddings_file = os.path.join(args.model_dir, "wordpiece_embeddings_teacher_h{}.pt".format(args.embedding_dimensionality))
            token_embedding_state_dict = torch.load(args.wordpiece_embeddings_file, map_location=args.device)
            embeddings_to_load = {"embeddings.weight": token_embedding_state_dict[teacher_token_embedding_name]}
            args.vocab_size = token_embedding_state_dict[teacher_token_embedding_name].shape[0]
            # if not os.path.exists(embeddings_file):
            #     raise ValueError("{} doesn't exist!".format(embeddings_file))
                # teacher_state_dict = torch.load(os.path.join(args.teacher_name, "pytorch_model.bin"), map_location=torch.device("cpu"))
                # embedding_weights = teacher_state_dict[teacher_token_embedding_name]
                # token_embedding_state_dict = {teacher_token_embedding_name: embedding_weights}
                # assert args.token_embedding_dimensionality == embedding_weights.shape[-1]
                # torch.save(token_embedding_state_dict, embeddings_file)
            # else:
            # for token_embedding_name in token_embedding_names:
            #     embeddings_to_load[token_embedding_name] = token_embedding_state_dict[teacher_token_embedding_name]

        # retrieve learned word embeddings, e.g. word2vec
        elif args.embedding_type == "word":
            logger.info("Initialising word embedding parameters from learned word vectors")
            word_embeddings = torch.load(args.processed_word_vectors_file, map_location=args.device)
            embeddings_to_load = {"embeddings.weight": word_embeddings}
            args.vocab_size = word_embeddings.shape[0]

            # add to the word vectors randomly initialised embeddings for any additional special tokens
            # add special token embeddings for BERT
            # if args.student_type == "BERT":
            #     word_embeddings = torch.cat((word_embeddings, 
            #                                  torch.FloatTensor(len(tokenizer.added_tokens_encoder), word_embeddings.shape[1])\
            #                                  .uniform_(-0.25, 0.25).to(args.device)))
            # for token_embedding_name in token_embedding_names:
            #     embeddings_to_load[token_embedding_name] = word_embeddings
        else:
            raise ValueError("Embedding type '' not recognised.".format(args.embedding_type))
        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_dimensionality, padding_idx=args.padding_idx)
        # print(self)
        # print(embeddings_to_load)

        param_keys = self.load_state_dict(embeddings_to_load, strict=True)
        loaded_params = [p for p in self.state_dict() if p not in param_keys[0]]
        num_loaded_params = sum([self.state_dict()[p].numel() for p in loaded_params])
        num_all_params = sum([p.numel() for n, p in self.state_dict().items()])
        logger.info("Loaded {} parameters (out of total {}) into: {}".format(num_loaded_params, num_all_params, loaded_params))
        # self.to(args.device)
        # logger.info("Student model created.")

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        return embeddings
