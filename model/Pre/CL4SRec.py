import os
import random

import numpy as np
import torch
import torch.nn as nn

from base.seq_recommender import SequentialRecommender
from data.augmentor import SequenceAugmentor
from model.Module.SASRec_module import SASRec_Model
from util.conf import OptionConf
from util.loss_torch import InfoNCE, l2_reg_loss
from util.sampler import next_batch_sequence

from fisher import compute_fisher_experts_from_bert4rec, load_round1_and_enable_shared


torch.cuda.set_device(0)


class CL4SRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CL4SRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CL4SRec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        head_num = int(args['-n_heads'])
        self.aug_type = int(args['-aug_type'])
        self.aug_rate = float(args['-aug_rate'])
        self.cl_rate = float(args['-cl_rate'])
        self.strategy = int(args['-strategy'])
        datasetFile = self.config['dataset']
        self.round2 = int(args['-round2_enable_shared'])
        self.model = SASRec_Model(
            self.data, self.emb_size, self.max_len,
            block_num, head_num, drop_rate,
            self.feature, datasetFile, self.strategy
        )
        initializer = nn.init.xavier_uniform_
        self.model.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 2, self.emb_size)))
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.model_name = self.config['model.name']

    def train(self):
        model = self.model.cuda()
        if self.round2 == 1:
            load_round1_and_enable_shared(
                model,
                '/root/autodl-tmp/fisher_cache/fisher_experts_epoch_cl4_beer20.pt',
                enable_shared=True
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        local_random = random.Random(2024)

        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(
                next_batch_sequence(self.data, self.batch_size, self.labelgap, max_len=self.max_len)
            ):
                seq, seqfull, pos, posfull, y, neg_idx, seq_len, popgap = batch
                seq_emb = model.forward(seq, pos)

                switch = local_random.sample(range(3), k=2)

                if switch[0] == 0:
                    aug_seq1, aug_pos1, aug_len1 = SequenceAugmentor.item_crop(seq, seq_len, self.aug_rate)
                    aug_emb1 = model.forward(aug_seq1, aug_pos1)
                    cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(aug_len1)]
                elif switch[0] == 1:
                    aug_seq1 = SequenceAugmentor.item_reorder(seq, seq_len, self.aug_rate)
                    aug_emb1 = model.forward(aug_seq1, pos)
                    cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                else:
                    aug_seq1 = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate, self.data.item_num + 1)
                    aug_emb1 = model.forward(aug_seq1, pos)
                    cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]

                if switch[0] == 0:
                    aug_seq2, aug_pos2, aug_len2 = SequenceAugmentor.item_crop(seq, seq_len, self.aug_rate)
                    aug_emb2 = model.forward(aug_seq2, aug_pos2)
                    cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(aug_len2)]
                elif switch[0] == 1:
                    aug_seq2 = SequenceAugmentor.item_reorder(seq, seq_len, self.aug_rate)
                    aug_emb2 = model.forward(aug_seq2, pos)
                    cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                else:
                    aug_seq2 = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate, self.data.item_num + 1)
                    aug_emb2 = model.forward(aug_seq2, pos)
                    cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]

                cl_loss = self.cl_rate * InfoNCE(torch.cat(cl_emb1, 0), torch.cat(cl_emb2, 0), 1, True)
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss + l2_reg_loss(self.reg, model.item_emb) + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'rec_loss:', rec_loss.item())

            model.eval()
            self.fast_evaluation(epoch, self.data1)

        save_path = f"./fisher_cache/fisher_experts_epoch_cl4_beer{self.maxEpoch}.pt"
        compute_fisher_experts_from_bert4rec(
            bert4rec_obj=self,
            model=model,
            epoch=self.maxEpoch,
            save_path=save_path,
            model_name=self.model_name,
            max_batches=None,
            use_eval_mode=True,
            include_cl_loss=True,
            disable_noisy_gating=True
        )

    def calculate_loss(self, seq_emb, y, neg, pos):
        y = torch.tensor(y)
        neg = torch.tensor(neg)
        y_emb = self.model.item_emb[y]
        neg_emb = self.model.item_emb[neg]
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)
        pos_labels = torch.ones(pos_logits.shape).cuda()
        neg_labels = torch.zeros(neg_logits.shape).cuda()
        indices = np.where(pos != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def predict(self, seq, pos, seq_len, gap):
        with torch.no_grad():
            seq_emb = self.model.forward(seq, pos)
            last_item_embeddings = [seq_emb[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
            score = torch.matmul(torch.cat(last_item_embeddings, 0), self.model.item_emb.transpose(0, 1))
        return score.cpu().numpy()
