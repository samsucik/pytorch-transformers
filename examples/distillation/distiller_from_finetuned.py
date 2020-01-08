# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" The distiller to distil DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import os
import re
import math
import psutil
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import socket
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adadelta
from torch.utils.data import DataLoader

from pytorch_transformers import WarmupLinearSchedule, WarmupConstantSchedule
from examples.distillation.utils import logger
from examples.run_glue import set_seed, evaluate

def human_format(num):
    # from https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{:.2f}{}".format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

class Distiller:
    def __init__(
        self,
        params,
        dataset_train: DataLoader,
        dataset_eval: DataLoader,
        student: nn.Module,
        evaluate_fn=None,
        student_type="BERT" # one of BERT, LSTM
    ):
        logger.info('Initializing Distiller')
        assert student_type in ["BERT", "LSTM"]
        self.student_type = student_type
        self.params = params
        self.output_dir = params.output_dir
        self.student = student
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.evaluate_fn = evaluate_fn
        self.num_steps_epoch = len(self.dataset_train)
        self.get_data_iterator()
        self.temperature = params.temperature
        assert self.temperature > 0.
        self.use_hard_labels = params.use_hard_labels

        self.alpha_ce = params.alpha_ce
        self.alpha_mse = params.alpha_mse
        assert self.alpha_ce >= 0.
        assert self.alpha_mse >= 0.
        assert self.alpha_ce + self.alpha_mse > 0.
        self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss_fct = nn.MSELoss(reduction='sum') # 'batchmean' achieved as 'sum' + manual normalising
        self.ce_simple_loss_fct = nn.CrossEntropyLoss(reduction='mean')

        self.n_epochs = params.n_epochs
        self.epoch = 0 # epoch counter; TODO set to non-zero when resuming distillation from checkpoint
        self.n_iter = 0 # within-epoch step counter (step = parameters update)
        self.n_total_iter = 0 # overall step counter
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.best_dev_score = -100000
        self.best_dev_score_epoch = 0.0 # can be epoch fractions because evaluation is done many times per epoch
        self.last_log_time = 0

        logger.info('--- Initializing model optimizer')
        assert params.gradient_accumulation_steps >= 1
        num_train_optimization_steps = int(self.num_steps_epoch / params.gradient_accumulation_steps * self.n_epochs) + 1

        if self.student_type == "BERT":
            no_decay = ['bias', 'LayerNorm.weight']
            student_params = [
                {'params': [p for n, p in self.student.named_parameters() 
                    if not any(nd in n for nd in no_decay) and p.requires_grad], 
                 'weight_decay': params.weight_decay},
                {'params': [p for n, p in self.student.named_parameters() 
                    if any(nd in n for nd in no_decay) and p.requires_grad], 
                 'weight_decay': 0.0}
            ]
            self.parameters_to_clip = self.student.parameters()
        else: # no weight decay for LSTM student
            student_params = [p for p in self.student.parameters() if p.requires_grad]
            self.parameters_to_clip = self.student.non_embedding_params()

        num_params = sum([p.numel() for p in self.student.parameters()])
        self.params.n_params = num_params
        logger.info("------ Number of all parameters: {} ({})".format(num_params, human_format(num_params)))
        
        num_trainable_params = sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        self.params.n_params_train = num_trainable_params
        logger.info("------ Number of trainable parameters (all): {} ({})".format(num_trainable_params, human_format(num_trainable_params)))

        num_trainable_params_embed = sum([p.numel() for n, p in self.student.named_parameters() if (p.requires_grad and "embed" in n)])
        self.params.n_params_train_embed = num_trainable_params_embed
        logger.info("------ Number of embedding trainable parameters: {} ({})".format(num_trainable_params_embed, human_format(num_trainable_params_embed)))
        
        num_trainable_params_other = num_trainable_params - num_trainable_params_embed
        self.params.n_params_train_other = num_trainable_params_other
        logger.info("------ Number of other trainable parameters: {} ({})".format(num_trainable_params_other, human_format(num_trainable_params_other)))
        
        if params.optimizer == "adam":
            self.optimizer = AdamW(student_params, lr=params.learning_rate, eps=1e-6, betas=(0.9, 0.98))
            logger.info("------ Using Adam optimizer.")
        elif params.optimizer == "adadelta":
            self.optimizer = Adadelta(student_params, lr=params.learning_rate, rho=0.95)
            logger.info("------ Using ADADELTA optimizer.")
        else:
            raise ValueError("Unrecognised optimizer option: {}".format(params.optimizer))

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = WarmupConstantSchedule(self.optimizer, warmup_steps)

        logger.info('--- Initializing Tensorboard')
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(self.params.output_dir, "tb_{}_{}".format(current_time, socket.gethostname()))
        self.tensorboard = SummaryWriter(log_dir=logdir, flush_secs=60)
        self.tensorboard.add_text(tag='config', text_string=str(self.params), global_step=0)

    def get_data_iterator(self):
        """
        Initialize the data iterator.
        Each process has its own data iterator (iterating on his own random portion of the dataset).
        """
        logger.info("--- Initializing Data Iterator")
        set_seed(self.params)
        # if self.student_type == "LSTM": self.dataset_train.init_epoch()
        self.data_iterator = tqdm(self.dataset_train, desc="Iteration", total=(self.params.max_steps % self.num_steps_epoch 
            if self.params.max_steps > 0 else None))

    def train(self):
        """
        The real training loop.
        """
        logger.info('Starting training')
        self.last_log_time = time.time()
        self.student.train()
        do_stop = False
        for epoch_number in range(self.n_epochs):
            logger.info("--- Starting epoch {}/{}".format(self.epoch, self.n_epochs-1))
            for step, batch in enumerate(self.data_iterator):
                self.step(batch)

                if self.n_total_iter % self.params.log_interval == 0 and self.params.evaluate_during_training:
                    eval_params = SimpleNamespace(dataset=self.dataset_eval, model=self.student, student=self.student_type, \
                                                  task_name=self.params.task_name, device=self.params.device)
                    results = self.evaluate_fn(eval_params)            
                    dev_score_tuple = [(name, score) for name, score in results.items() if name != "additional"][0]
                    dev_score = dev_score_tuple[1]
                    self.tensorboard.add_scalar('eval_{}'.format(dev_score_tuple[0]), dev_score, global_step=self.n_total_iter)
                    logger.info("Dev {}: {}".format(dev_score_tuple[0], dev_score))
                    if "additional" in results:
                        for name, val in results["additional"]:
                            logger.info("Dev {}: {}".format(name, val))

                    # save best checkpoint
                    if self.best_dev_score < dev_score:
                        self.best_dev_score_epoch = self.n_total_iter/self.num_steps_epoch
                        self.best_dev_score = dev_score
                        self.save_checkpoint("e{:.2f}_{}:{:.3f}".format(self.best_dev_score_epoch, dev_score_tuple[0], dev_score), kind="best")
                    
                    self.student.train()

                if self.params.max_steps > 0 and self.n_total_iter + step > self.params.max_steps:
                    self.data_iterator.close()
                    do_stop = True
                    break

            if do_stop:
                logger.info("--- Ending epoch {}/{} early due to max_steps".format(self.epoch, self.n_epochs-1))
                self.end_epoch()
                break

            self.data_iterator.close()
            # if self.student_type == "LSTM": self.dataset_train.init_epoch()
            self.data_iterator = tqdm(self.dataset_train, desc="Iteration")
            
            logger.info("--- Ending epoch {}/{}".format(self.epoch, self.n_epochs-1))
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint()
        self.tensorboard.close()
        logger.info("Training is finished")

    def step(self, batch):
        """
        One optimization step: forward of student, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).
        """
        if self.student_type == "BERT":
            batch = tuple(t.to(self.params.device) for t in batch)
            logits = self.student(input_ids=batch[0])[0]
            n_sequences = batch[0].size(0)
        else:
            # TODO: convert batch to this device?
            logits = self.student((batch[0], batch[2])) # batch[0] = sentence, batch[2] = sentence length
            n_sequences = batch[0].size(0)

        if self.use_hard_labels: # currently broken for BERT as labels are not included in the transfer set
            # loss = self.ce_simple_loss_fct(logits, labels)
            raise Error("Using hard labels is not currently supported, you should set: use_hard_labels=False.")
        else:
            teacher_logits = batch[1]
            assert logits.size() == teacher_logits.size()

            loss_ce = self.alpha_ce * self.ce_loss_fct(F.log_softmax(logits/self.temperature, dim=-1),
                F.softmax(teacher_logits/self.temperature, dim=-1)) * (self.temperature)**2
            loss_mse = self.alpha_mse * self.mse_loss_fct(logits, teacher_logits)/logits.size(0) # Reproducing batchmean reduction
            loss = loss_mse + loss_ce
                
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()        
        self.optimize(loss)
        self.n_sequences_epoch += n_sequences

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error('NaN detected')
            exit()

        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(self.parameters_to_clip, self.params.max_grad_norm)
            self.tensorboard.add_scalar(tag="grad_norm", scalar_value=grad_norm, global_step=self.n_total_iter)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log_time = time.time()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        self.tensorboard.add_scalar(tag="losses/cum_avg_loss_epoch", scalar_value=self.total_loss_epoch/self.n_iter, 
            global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], 
            global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="global/memory_usage", scalar_value=psutil.virtual_memory()._asdict()['used']/1_000_000, 
            global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="global/speed", scalar_value=time.time()-self.last_log_time, 
            global_step=self.n_total_iter)

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f'{self.n_sequences_epoch} sequences have been trained during this epoch.')

        if (self.params.checkpoint_interval > 1 and self.epoch > 0 and self.epoch % self.params.checkpoint_interval == 0) or \
           (self.params.checkpoint_interval == 1):
            self.save_checkpoint(checkpoint_name=f'model_epoch_{self.epoch}.pth')
        
        self.tensorboard.add_scalar(tag='epoch/loss', scalar_value=self.total_loss_epoch/self.n_iter, global_step=self.epoch)

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name=None, kind=None):
        """
        Save the current state. Only by the master process.
        """

        # delete all previous best checkpoints
        if kind == "best":
            checkpoint_name = ("best_" + checkpoint_name) if checkpoint_name is not None else "best"
            previous_bests = [f for f in os.listdir(self.output_dir) if re.match(r'.*best.*\.bin', f)]
            logger.info("Deleting previous best checkpoint(s): {}".format(previous_bests))
            for f in previous_bests: os.remove(os.path.join(self.output_dir, f))

        if checkpoint_name is not None:
            checkpoint_name = "pytorch_model_" + checkpoint_name + ".bin"

        mdl_to_save = self.student.module if hasattr(self.student, 'module') else self.student
        if checkpoint_name is not None:
            if self.student_type == "BERT": mdl_to_save.config.save_pretrained(self.output_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(self.output_dir, checkpoint_name))
        else:
            if self.student_type == "BERT":
                mdl_to_save.save_pretrained(self.output_dir)
            else:
                state_dict = mdl_to_save.state_dict()
                torch.save(state_dict, os.path.join(self.output_dir, "pytorch_model.bin"))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.params, os.path.join(self.output_dir, 'training_args.bin'))
        