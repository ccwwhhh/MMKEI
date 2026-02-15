import math
import os
import random
from datetime import datetime
from math import floor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from base.seq_recommender import SequentialRecommender
from model.Module.LinRec_module_bert import LinRec_Model_bert
from util.conf import OptionConf
from util.loss_torch import InfoNCE
from util.sampler import next_batch_sequence

from fisher import compute_fisher_experts_from_bert4rec, load_round1_and_enable_shared


torch.cuda.set_device(0)


class BERTLin(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BERTLin, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['BERTLin'])
        datasetFile = self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type = args['-cltype']
        self.cl = float(args['-cl'])
        head_num = int(args['-n_heads'])
        self.strategy = float(args['-strategy'])
        self.aug_rate = float(args['-mask_rate'])
        self.round2 = int(args['-round2_enable_shared'])
        self.eps = float(args['-eps'])
        self.model = LinRec_Model_bert(
            self.data, self.emb_size, self.max_len,
            block_num, head_num, drop_rate,
            self.feature, datasetFile, self.strategy
        )
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.model_name = self.config['model.name']

        with open("./item-popular/count_pantry.txt", 'r') as file:
            data = None
            for line in file:
                line = line.strip()
                if line:
                    arr = line[1:-1].split(", ")
                    data = np.asfarray(arr, float)
        self.data1 = data

    def train(self):
        model = self.model.cuda()
        if self.round2 == 1:
            load_round1_and_enable_shared(
                model,
                '/root/autodl-tmp/fisher_cache/fisher_experts_epoch_bertlin_beer20.pt',
                enable_shared=True
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            self.listcountitem = [0] * (self.data.item_num + 2)
            model.train()

            for n, batch in enumerate(
                next_batch_sequence(self.data, self.batch_size, self.labelgap, max_len=self.max_len)
            ):
                seq, seqfull, pos, posfull, y, neg_idx, seq_len, labelgap = batch

                aug_seq, masked, labels = self.item_mask_for_bert(
                    seq, seq_len, self.aug_rate, self.data.item_num + 1
                )

                self.listcountitem = np.sum(
                    [self.count_tensor_elements(seq, self.data.item_num), self.listcountitem],
                    axis=0
                )

                seq_emb = model.forward(aug_seq, pos)

                cl_loss = self.cl_rate * self.cal_cl_loss(labels, seq_emb, masked) if self.cl == 1 else 0.0
                rec_loss = self.calculate_loss(seq_emb, masked, labels)
                batch_loss = cl_loss + rec_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())

            model.eval()
            self.fast_evaluation(epoch, self.data1)

        save_path = f"./fisher_cache/fisher_experts_epoch_bertlin_beer{self.maxEpoch}.pt"
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

    def item_mask_for_bert(self, seq, seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i] * mask_ratio), 1))
            masked[i, to_be_masked] = 1
            labels = labels + list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq, masked, np.array(labels)

    def calculate_loss(self, seq_emb, masked, labels):
        masked_t = torch.tensor(masked)
        seq_emb = seq_emb[masked_t > 0]
        seq_emb = seq_emb.view(-1, self.emb_size)

        if self.feature == 'text':
            emb = self.model.mlps(self.model.bert_tensor)
        elif self.feature == 'id':
            emb = self.model.item_emb
        elif self.feature == 'id+text':
            emb = self.model.item_emb + self.model.mlps(self.model.bert_tensor)
        else:
            raise ValueError(f"Unknown feature mode: {self.feature}")

        logits = torch.mm(seq_emb, emb.t())
        loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())
        return loss

    def predict(self, seq, pos, seq_len, gap_batch):
        with torch.no_grad():
            for i, length in enumerate(seq_len):
                if length == self.max_len:
                    seq[i, :length - 1] = seq[i, 1:]
                    pos[i, :length - 1] = pos[i, 1:]
                    pos[i, length - 1] = length
                    seq[i, length - 1] = self.data.item_num + 1
                else:
                    pos[i, length] = length + 1
                    seq[i, length] = self.data.item_num + 1

            seq_emb = self.model.forward(seq, pos)
            last_item_embeddings = [seq_emb[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]

            item_emb = self.model.item_emb
            if self.feature == 'text':
                item_emb = self.model.mlps(self.model.bert_tensor)
            elif self.feature == 'id+text':
                item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb

            score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def cal_cl_loss(self, label, seq_emb, masked):
        label = torch.tensor(label)
        user_view = seq_emb[torch.tensor(masked) > 0]

        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        elif self.feature == 'id+text':
            if self.cl_type == 'id':
                item_view = self.model.item_emb
            elif self.cl_type == 'text':
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb

        random_noise1 = torch.rand_like(item_view).cuda()
        random_noise2 = torch.rand_like(item_view).cuda()
        item_view_1 = item_view + torch.sign(item_view) * F.normalize(random_noise1, dim=-1) * self.eps
        item_view_2 = item_view + torch.sign(item_view) * F.normalize(random_noise2, dim=-1) * self.eps

        return InfoNCE(item_view_1[label], item_view_2[label], 0.2)

    def draw(self):
        ItemInd = [i for i in range(self.data.item_num)]
        ItemInd = random.sample(ItemInd, self.data.item_num)

        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        elif self.feature == 'id+text':
            if self.cl_type == 'id':
                item_view = self.model.item_emb
            elif self.cl_type == 'text':
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb

        item_view = item_view[1:]
        Pi = item_view.cpu().detach().numpy()

        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        sns.set_theme(style="white")
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')

        Pi = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi)

        for i in range(len(Pi)):
            k = math.sqrt(Pi[i][0] * Pi[i][0] + Pi[i][1] * Pi[i][1])
            Pi[i][0] = (Pi[i][0] / k)
            Pi[i][1] = (Pi[i][1] / k)

        columns = [' ', '  ']
        Pi = pd.DataFrame(Pi, columns=columns)
        sns.jointplot(x=' ', y='  ', data=Pi, kind="kde", cmap="Blues", shade=True, shade_lowest=True)

        plt.title("BERT4Rec+ID", y=-0.17, fontsize=20, weight='bold')
        plt.show()
        plt.savefig('./picture/BERT4Rec/fig' + str(datetime.now()) + '.svg', dpi=300, bbox_inches='tight', format="svg")
        plt.close()

    def count_tensor_elements(self, tensor, max_value):
        count_list = [0] * (max_value + 2)
        for element in tensor.reshape(-1):
            count_list[int(element.item())] += 1
        return count_list


def multiply_tensor_elements(tensor):
    result = []
    for i in range(len(tensor)):
        for j in range(i + 1, len(tensor)):
            result.append(tensor[i] * tensor[j])
    return torch.tensor(result)
