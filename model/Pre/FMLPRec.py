import math
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from base.seq_recommender import SequentialRecommender
from model.Module.FMLPRec_module import FMLPRecModel
from util.conf import OptionConf
from util.loss_torch import InfoNCE
from util.sampler import next_batch_sequence

from fisher import compute_fisher_experts_from_bert4rec, load_round1_and_enable_shared


torch.cuda.set_device(0)


class FMLPRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(FMLPRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['FMLPRec'])
        datasetFile = self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type = args['-cltype']
        self.cl = float(args['-cl'])
        head_num = int(args['-n_heads'])
        self.round2 = int(args['-round2_enable_shared'])
        self.model = FMLPRecModel(
            self.data, self.emb_size, self.max_len,
            block_num, head_num, drop_rate,
            self.feature, datasetFile
        )
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.eps = float(args['-eps'])
        self.model_name = self.config['model.name']

    def train(self):
        model = self.model.cuda()
        if self.round2 == 1:
            load_round1_and_enable_shared(
                model,
                '/root/autodl-tmp/fisher_cache/fisher_experts_epoch_FMLP_beer20.pt',
                enable_shared=True
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(
                next_batch_sequence(self.data, self.batch_size, self.labelgap, max_len=self.max_len)
            ):
                seq, seqfull, pos, posfull, y, neg_idx, _, gap = batch
                seq_emb = model.forward(seq, pos)

                cl_loss = self.cl_rate * self.cal_cl_loss(y, pos) if self.cl == 1 else 0.0
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 50 == 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'rec_loss:', float(rec_loss.item()),
                        'cl_loss:', float(cl_loss) if isinstance(cl_loss, (int, float)) else float(cl_loss.item()),
                    )

            model.eval()
            self.fast_evaluation(epoch, self.data1)

        save_path = f"./fisher_cache/fisher_experts_epoch_FMLP_beer{self.maxEpoch}.pt"
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

        if self.feature == 'text':
            y_emb = self.model.bert_tensor[y.cuda()]
            neg_emb = self.model.bert_tensor[neg.cuda()]
        elif self.feature == 'id':
            y_emb = self.model.item_emb[y]
            neg_emb = self.model.item_emb[neg]
        elif self.feature == 'id+text':
            y_emb = self.model.item_emb[y] + self.model.mlps(self.model.bert_tensor[y.cuda()])
            neg_emb = self.model.item_emb[neg] + self.model.mlps(self.model.bert_tensor[neg.cuda()])
        else:
            raise ValueError(f"Unknown feature mode: {self.feature}")

        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)
        pos_labels = torch.ones_like(pos_logits).cuda()
        neg_labels = torch.zeros_like(neg_logits).cuda()
        indices = np.where(pos != 0)

        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def predict(self, seq, pos, seq_len, gap):
        with torch.no_grad():
            seq_emb = self.model.forward(seq, pos)
            last_item_embeddings = [seq_emb[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
            item_emb = self.model.item_emb
            if self.feature == 'text':
                item_emb = self.model.bert_tensor
            elif self.feature == 'id+text':
                item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def cal_cl_loss(self, y, pos):
        y = torch.tensor(y)
        label = y[np.where(pos != 0)]

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

        plt.title("SASRec+ID", y=-0.17, fontsize=20, weight='bold')
        plt.show()
        plt.savefig('./picture/fig' + str(datetime.now()) + '.svg', dpi=300, bbox_inches='tight', format="svg")
        plt.close()

    def count_tensor_elements(self, tensor, max_value):
        count_list = [0] * (max_value + 1)
        for element in tensor.reshape(-1):
            count_list[int(element.item())] += 1
        return count_list
