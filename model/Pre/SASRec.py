import os
import random
from datetime import datetime
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.loss_torch import InfoNCE
from model.Module.SASRec_module import SASRec_Model

from fisher import compute_fisher_experts_from_bert4rec, load_round1_and_enable_shared


torch.cuda.set_device(0)


class SASRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SASRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SASRec'])
        datasetFile = self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type = args['-cltype']
        self.cl = float(args['-cl'])
        head_num = int(args['-n_heads'])
        self.strategy = float(args['-strategy'])
        self.model = SASRec_Model(
            self.data, self.emb_size, self.max_len,
            block_num, head_num, drop_rate,
            self.feature, datasetFile, self.strategy
        )
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.round2 = int(args['-round2_enable_shared'])
        self.eps = float(args['-eps'])
        self.listcountitem = [0] * (self.data.item_num + 1)
        self.model_name = self.config['model.name']

    def train(self):
        model = self.model.cuda()
        if self.round2 == 1:
            load_round1_and_enable_shared(
                model,
                '/root/autodl-tmp/fisher_cache/fisher_experts_epoch_sas_beer20.pt',
                enable_shared=True
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            self.listcountitem = [0] * (self.data.item_num + 1)
            model.train()

            for n, batch in enumerate(
                next_batch_sequence(self.data, self.batch_size, self.labelgap, max_len=self.max_len)
            ):
                seq, seqfull, pos, posfull, y, neg_idx, _, gap = batch
                self.listcountitem = np.sum(
                    [self.count_tensor_elements(seq, self.data.item_num), self.listcountitem],
                    axis=0
                ).tolist()

                seq_emb = model.forward(seq, pos)

                cl_loss = self.cl_rate * self.cal_cl_loss(y, pos) if self.cl == 1 else 0.0
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())

            model.eval()
            self.fast_evaluation(epoch, self.data1)

        save_path = f"./fisher_cache/fisher_experts_epoch_sas_beer{self.maxEpoch}.pt"
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
            y_emb = self.model.mlps(self.model.bert_tensor[y.cuda()])
            neg_emb = self.model.mlps(self.model.bert_tensor[neg.cuda()])
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
                item_emb = self.model.mlps(self.model.bert_tensor)
            elif self.feature == 'id+text':
                item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb

            score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def cal_cl_loss(self, y, pos):
        y = torch.tensor(y)
        label = y[np.where(pos != 0)]

        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.bert_tensor
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

    def drawtsne(self):
        ItemInd = [i for i in range(1, int(self.data.item_num))]

        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        sns.set_theme(style="white")
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')

        item_view2 = self.model.mlps(self.model.bert_tensor)
        item_view2 = item_view2[list(np.where(self.data1 >= 8)[0])]
        ItemInd2 = random.sample([i for i in range(0, len(item_view2))], int(len(item_view2) / 10))

        Pi2 = item_view2.cpu().detach().numpy()
        Pi2 = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi2)
        x2 = np.array(Pi2[ItemInd2, 0])
        y2 = np.array(Pi2[ItemInd2, 1])
        plt.scatter(x2, y2, c='blue', alpha=0.5, s=170)

        item_view1 = self.model.mlps(self.model.bert_tensor)[1:]
        Pi1 = item_view1.cpu().detach().numpy()
        ItemInd1 = random.sample(ItemInd, int((self.data.item_num) / 10))
        Pi1 = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi1)
        x1 = np.array(Pi1[ItemInd1, 0])
        y1 = np.array(Pi1[ItemInd1, 1])
        plt.scatter(x1, y1, c='red', alpha=0.5, s=170)

        plt.title("SASRec", y=-0.17, fontsize=20, weight='bold')
        plt.show()
        plt.savefig('./picture/fig' + str(datetime.now()) + '.svg', dpi=300, bbox_inches='tight', format="svg")
        plt.close()

    def draw(self):
        ItemInd = [i for i in range(1, self.data.item_num + 1)]
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

        item_view1 = item_view[1:]
        Pi = item_view1.cpu().detach().numpy()

        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        sns.set_theme(style="white")
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')

        Pi = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi)
        for i in range(len(Pi)):
            k = math.sqrt(Pi[i][0] * Pi[i][0] + Pi[i][1] * Pi[i][1])
            Pi[i][0] = Pi[i][0] / k
            Pi[i][1] = Pi[i][1] / k

        columns = [' ', '  ']
        Pi = pd.DataFrame(Pi, columns=columns)
        sns.jointplot(x=' ', y='  ', data=Pi, kind="kde", cmap="Blues", shade=True, shade_lowest=True)

        plt.title("SASRec", y=-0.17, fontsize=20, weight='bold')
        plt.show()
        plt.savefig('./picture/fig' + str(datetime.now()) + '.svg', dpi=300, bbox_inches='tight', format="svg")
        plt.close()

    def find_popular_num(self):
        list_popular = list(self.data1)
        num_to_select_updated = max(1, int(len(list_popular) * 0.4))
        top_indices_updated = sorted(range(len(list_popular)), key=lambda i: list_popular[i], reverse=True)[
                              :num_to_select_updated]
        top_popularity_updated = [list_popular[index] for index in top_indices_updated]
        del top_indices_updated[0]
        boundary_popularity = min(top_popularity_updated)
        return boundary_popularity, top_indices_updated

    def count_tensor_elements(self, tensor, max_value):
        count_list = [0] * (max_value + 1)
        for element in tensor.reshape(-1):
            count_list[int(element.item())] += 1
        return count_list

    def count_distance_2(self):
        if self.feature == 'text':
            epoch_total = self.model.mlps(self.model.bert_tensor)[1:]
        elif self.feature == 'id':
            epoch_total = self.model.item_emb[1:]
        else:
            return

        _, top_index = self.find_popular_num()
        top_index = [*map(lambda x: x - 1, top_index)]
        epoch_pop = epoch_total[top_index]

        mask = torch.ones(epoch_total.size(0), dtype=torch.bool)
        mask[top_index] = False
        epoch_other = epoch_total[mask]

        sample_pop = random.sample(list(range(0, len(epoch_pop))), max(1, len(epoch_pop) // 2))
        emb_selectpop = epoch_pop[sample_pop].reshape([-1, 64])
        emb_selectpop = F.normalize(emb_selectpop, dim=-1)
        dist_pop = torch.pdist(emb_selectpop, p=2).mean()
        print('dist_pop', dist_pop)

        sample_other = random.sample(list(range(0, len(epoch_other))), max(1, len(epoch_other) // 10))
        emb_selectother = epoch_other[sample_other].reshape([-1, 64])
        emb_selectother = F.normalize(emb_selectother, dim=-1)
        dist_other = torch.pdist(emb_selectother, p=2).mean()
        print('dist_other', dist_other)
