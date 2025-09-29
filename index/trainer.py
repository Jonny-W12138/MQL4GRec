import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from utils import ensure_dir,set_color,get_local_time
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        # saved_model_dir = "{}".format(get_local_time())
        # self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        
        # wandb设置
        self.use_wandb = getattr(args, 'use_wandb', False)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_rq_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_rq_loss += rq_loss

        return total_loss, total_recon_loss, total_rq_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        # indices_set = set()
        indices_list = []
        num_sample = 0
        # 用于统计每个码本层的使用情况
        codebook_usage = [set() for _ in range(len(self.args.num_emb_list))]
        
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices, distances = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            
            # 统计每个码本层的使用情况
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_list.append(code)
                
                # 记录每个码本层使用的索引
                for layer_idx, idx in enumerate(index):
                    codebook_usage[layer_idx].add(int(idx))
        
        freq_count = defaultdict(int)      
        for c in indices_list:
            freq_count[c] += 1
        max_value = max(list(freq_count.values()))
        min_value = min(list(freq_count.values()))

        indices_set = set(indices_list)
        collision_rate = (num_sample - len(indices_set))/num_sample
        
        # 计算码本利用率
        codebook_utilization = []
        total_utilization = 0
        for layer_idx, used_indices in enumerate(codebook_usage):
            total_codes = self.args.num_emb_list[layer_idx]
            used_codes = len(used_indices)
            utilization = used_codes / total_codes
            codebook_utilization.append(utilization)
            total_utilization += utilization
        
        avg_utilization = total_utilization / len(codebook_utilization)

        return max_value, min_value, collision_rate, codebook_utilization, avg_utilization

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, total_rq_loss = self._train_epoch(data, epoch_idx)
            print(f'epoch {epoch_idx}, total loss: {train_loss}, recon loss: {train_recon_loss}, rq loss: {total_rq_loss}')
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)
            
            # 记录训练指标到wandb
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch_idx,
                    "train/total_loss": train_loss,
                    "train/recon_loss": train_recon_loss,
                    "train/rq_loss": total_rq_loss,
                    "train/training_time": training_end_time - training_start_time
                })

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                max_value, min_value, collision_rate, codebook_utilization, avg_utilization = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                print(f'MAX: {max_value}, Min: {min_value}, collision_rate: {collision_rate}, best_collision_rate: {self.best_collision_rate}')
                print(f'Codebook utilization: {[f"{u:.4f}" for u in codebook_utilization]}, Avg: {avg_utilization:.4f}')

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f, "
                    + set_color("avg_utilization", "blue")
                    + ": %.4f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate, avg_utilization)

                self.logger.info(valid_score_output)
                
                # 记录验证指标到wandb
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb_log_dict = {
                        "epoch": epoch_idx,
                        "eval/max_frequency": max_value,
                        "eval/min_frequency": min_value,
                        "eval/collision_rate": collision_rate,
                        "eval/best_collision_rate": self.best_collision_rate,
                        "eval/avg_codebook_utilization": avg_utilization,
                        "eval/validation_time": valid_end_time - valid_start_time
                    }
                    
                    # 记录每个码本层的利用率
                    for i, util in enumerate(codebook_utilization):
                        wandb_log_dict[f"eval/codebook_utilization_layer_{i}"] = util
                    
                    wandb.log(wandb_log_dict)
                
                # if epoch_idx > 1000:
                # if epoch_idx % 10 == 0:
                #     self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate




