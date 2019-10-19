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
import math
import psutil
import time
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
import numpy as np
import psutil
import socket
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from pytorch_transformers import WarmupLinearSchedule

from examples.distillation.utils import logger
from examples.distillation.dataset import Dataset
from examples.run_glue import set_seed, evaluate

class Distiller:
    def __init__(self,
                 params: dict,
                 dataloader: Dataset,
                 student: nn.Module,
                 teacher: nn.Module,
                 tokenizer: nn.Module):
        logger.info('Initializing Distiller')
        self.params = params
        self.output_dir = params.output_dir
        # self.multi_gpu = params.multi_gpu

        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer

        self.dataloader = dataloader
        # if self.params.n_gpu > 1:
        #     self.dataloader.split()
        # self.num_steps_epoch = int(len(self.dataloader) / params.batch_size) + 1
        self.num_steps_epoch = len(self.dataloader)
        # print(len(self.dataloader), params.batch_size)
        # print(self.num_steps_epoch, params.gradient_accumulation_steps, params.n_epoch)
        # exit(0)
        self.get_iterator()

        self.temperature = params.temperature
        assert self.temperature > 0.

        self.alpha_ce = params.alpha_ce
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        assert self.alpha_ce >= 0.
        assert self.alpha_mse >= 0.
        assert self.alpha_cos >= 0.
        assert self.alpha_ce + self.alpha_mse + self.alpha_cos > 0.

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        if self.alpha_mse > 0.: self.last_loss_mse = 0
        if self.alpha_cos > 0.: self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
        if self.alpha_mse > 0.:
            self.mse_loss_fct = nn.MSELoss(reduction='sum')
        if self.alpha_cos > 0.:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')

        logger.info('--- Initializing model optimizer')
        assert params.gradient_accumulation_steps >= 1
        num_train_optimization_steps = int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': params.weight_decay},
            {'params': [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        logger.info("------ Number of trainable parameters (student): %i" % sum([p.numel() for p in self.student.parameters() if p.requires_grad]))
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=params.learning_rate,
                               eps=params.adam_epsilon,
                               betas=(0.9, 0.98))

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = WarmupLinearSchedule(self.optimizer,
                                                warmup_steps=warmup_steps,
                                                t_total=num_train_optimization_steps)
        # print("TOTAL", num_train_optimization_steps)
        # print("WARM UP", warmup_steps)
        # exit(0)

        # if self.multi_gpu:
        #     if self.fp16:
        #         from apex.parallel import DistributedDataParallel
        #         logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
        #         self.student = DistributedDataParallel(self.student)
        #     else:
        #         from torch.nn.parallel import DistributedDataParallel
        #         logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
        #         self.student = DistributedDataParallel(self.student,
        #                                                device_ids=[params.local_rank],
        #                                                output_device=params.local_rank)

        logger.info('--- Initializing Tensorboard')
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(self.params.output_dir, "tensorboard", current_time + '_' + socket.gethostname())
        self.tensorboard = SummaryWriter(log_dir=logdir, flush_secs=60)
        self.tensorboard.add_text(tag='config', text_string=str(self.params), global_step=0)

    def get_iterator(self):
        """
        Initialize the data iterator.
        Each process has its own data iterator (iterating on his own random portion of the dataset).
        """
        logger.info('--- Initializing Data Iterator')
        set_seed(self.params)
        self.data_iterator = tqdm(self.dataloader, desc="Iteration", disable=self.params.local_rank not in [-1, 0],
                                  total=(self.params.max_steps % self.num_steps_epoch if  self.params.max_steps > 0 else None))

    def train(self):
        """
        The real training loop.
        """
        logger.info('Starting training')
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()
        do_stop = False
        for epoch_number in range(self.params.n_epoch):
            logger.info(f'--- Starting epoch {self.epoch}/{self.params.n_epoch-1}')
            # if self.multi_gpu:
            #     torch.distributed.barrier()

            for step, batch in enumerate(self.data_iterator):
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
                batch = tuple(t.to(self.params.device) for t in batch)
                self.step(batch)

                # print(self.params.local_rank)
                # exit(0)
                if self.n_total_iter % self.params.log_interval == 0 and self.params.local_rank in [-1, 0] and self.params.evaluate_during_training:
                    results = evaluate(self.params, self.student, self.tokenizer, prefix="e{}s{}".format(epoch_number, step))
                    for key, value in results.items():
                        if key == "conf_mtrx": continue
                        self.tensorboard.add_scalar('eval_{}'.format(key), value, global_step=self.n_total_iter)

                # iter_bar.update()
                # iter_bar.set_postfix({'Last_loss': f'{self.last_loss:.2f}',
                #                       'Avg_cum_loss': f'{self.total_loss_epoch/self.n_iter:.2f}'})
                if self.params.max_steps > 0 and self.n_total_iter + step > self.params.max_steps:
                    self.data_iterator.close()
                    do_stop = True
                    break

            if do_stop:
                logger.info(f'--- Ending epoch {self.epoch}/{self.params.n_epoch-1} early due to max_steps')
                self.end_epoch()
                break

            self.data_iterator.close()
            self.data_iterator = tqdm(self.dataloader, desc="Iteration", disable=self.params.local_rank not in [-1, 0])
            
            logger.info(f'--- Ending epoch {self.epoch}/{self.params.n_epoch-1}')
            self.end_epoch()

        logger.info(f'Save very last checkpoint as `pytorch_model.bin`.')
        self.save_checkpoint()
        self.tensorboard.close()
        logger.info('Training is finished')

    def step(self, batch):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).
        """
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}
        
        (_, s_logits,) = self.student(**inputs)

        with torch.no_grad():
            (_, t_logits,) = self.teacher(**inputs)
        assert s_logits.size() == t_logits.size()

        loss_ce = self.ce_loss_fct(F.log_softmax(s_logits/self.temperature, dim=-1),
                                   F.softmax(t_logits/self.temperature, dim=-1)) * (self.temperature)**2
        loss = self.alpha_ce*loss_ce
        if self.alpha_mse > 0.:
            loss_mse = self.mse_loss_fct(s_logits, t_logits)/s_logits.size(0) # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse
        
        """
        if self.alpha_cos > 0.:
            s_hidden_states = s_hidden_states[-1]                              # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]                              # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)     # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)
            
            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)        # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)                # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)        # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)                # (bs * seq_length, dim)
        
            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1) # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos
        """

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mse > 0.:
            self.last_loss_mse = loss_mse.item()
        # if self.alpha_cos > 0.:
        #     self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += inputs["input_ids"].size(0)

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

        # if self.multi_gpu:
        #     loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
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
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """

        # for param_name, param in self.student.named_parameters():
        #     self.tensorboard.add_scalar(tag='parameter_mean/' + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter)
        #     self.tensorboard.add_scalar(tag='parameter_std/' + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter)
        #     if param.grad is None:
        #         continue
        #     self.tensorboard.add_scalar(tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(),global_step=self.n_total_iter)
        #     self.tensorboard.add_scalar(tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter)

        self.tensorboard.add_scalar(tag="losses/cum_avg_loss_epoch", scalar_value=self.total_loss_epoch/self.n_iter, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter)
        if self.alpha_mse > 0.:
            self.tensorboard.add_scalar(tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter)
        # if self.alpha_cos > 0.:
        #     self.tensorboard.add_scalar(tag="losses/loss_cos", scalar_value=self.last_loss_cos, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter)
        
        self.tensorboard.add_scalar(tag="global/memory_usage", scalar_value=psutil.virtual_memory()._asdict()['used']/1_000_000, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(tag="global/speed", scalar_value=time.time()-self.last_log, global_step=self.n_total_iter)

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f'{self.n_sequences_epoch} sequences have been trained during this epoch.')

        self.save_checkpoint(checkpoint_name=f'model_epoch_{self.epoch}.pth')
        self.tensorboard.add_scalar(tag='epoch/loss', scalar_value=self.total_loss_epoch/self.n_iter, global_step=self.epoch)

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name=None):
        """
        Save the current state. Only by the master process.
        """
        mdl_to_save = self.student.module if hasattr(self.student, 'module') else self.student
        if checkpoint_name is not None:
            mdl_to_save.config.save_pretrained(self.output_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(self.output_dir, checkpoint_name))
        else:
            mdl_to_save.save_pretrained(self.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.params, os.path.join(self.output_dir, 'training_args.bin'))
