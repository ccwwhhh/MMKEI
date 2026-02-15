import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
from data.pretrain import Pretrain

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig,\
    BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel, T5Tokenizer, T5Model, T5TokenizerFast
from math import sqrt
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
torch.cuda.set_device(0)
current_device = torch.cuda.current_device()


class BSARecModel(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(BSARecModel, self).__init__()


        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature

        self.datasetFile = datasetFile


        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(self.drop_rate)
        self.item_encoder = BSARecEncoder(self.block_num )
        self.apply(self.init_weights)

        initializer = nn.init.xavier_uniform_
        self.bert_config2 = BertConfig.from_pretrained('/root/autodl-tmp/bert')
        self.llm_model2 = BertModel.from_pretrained(
            '/root/autodl-tmp/bert',

            local_files_only=True,
            config=self.bert_config2,
        ).cuda()
        self.tokenizer2 = BertTokenizer.from_pretrained(
            '/root/autodl-tmp/bert')

        self.tokenizer3 = T5TokenizerFast.from_pretrained(
            '/root/autodl-tmp/P5-sportbase', legacy=False)
        self.llm_model3 = T5Model.from_pretrained(
            '/root/autodl-tmp/P5-sportbase',
            local_files_only=True,
        ).cuda()

        self.tokenizer4 = T5TokenizerFast.from_pretrained(
            '/root/autodl-tmp/P5-beautybase', legacy=False)
        self.llm_model4 = T5Model.from_pretrained(
            '/root/autodl-tmp/P5-beautybase',
            local_files_only=True,
        ).cuda()

        self.tokenizer6 = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/qwen3-0.6B",
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer6 = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/qwen3-0.6B",
            local_files_only=True,
            trust_remote_code=True,
        )

        self.llm_model6 = AutoModel.from_pretrained(
            "/root/autodl-tmp/qwen3-0.6B",
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).cuda()
        self.llama_config = LlamaConfig.from_pretrained('/root/autodl-tmp/LLaMA')
        self.llama_config.num_hidden_layers = 7
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.tokenizer5 = LlamaTokenizer.from_pretrained(
            '/root/autodl-tmp/LLaMA')
        self.llama = LlamaModel.from_pretrained(
            '/root/autodl-tmp/LLaMA',
            local_files_only=True,
            config=self.llama_config,
        ).cuda()
        self.llm_modelgpt = GPT2Model.from_pretrained('/root/autodl-tmp/gpt2').cuda()
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('/root/autodl-tmp/gpt2')


        self.word_embeddings2 = self.llm_model2.get_input_embeddings().weight
        self.word_embeddings3 = self.llm_model3.get_input_embeddings().weight
        self.word_embedding_gpt2 = self.llm_modelgpt.get_input_embeddings().weight
        self.word_embeddings4 = self.llm_model4.get_input_embeddings().weight
        self.word_embeddings5 = self.llama.get_input_embeddings().weight
        self.word_embedding_qwen = self.llm_model6.get_input_embeddings().weight
        self.word_embeddings4_1 = self.word_embeddings4.clone()
        self.word_embeddings6 = self.word_embeddings2.clone()
        self.word_embeddings7 = self.word_embeddings2.clone()
        self.word_embeddings8 = self.word_embeddings2.clone()
        self.word_embeddings3_1 = self.word_embeddings3.clone()


        def generate_random_param(shape):
            return nn.Parameter(torch.normal(mean=0.0, std=0.02, size=shape))


        random_5 = generate_random_param(self.word_embeddings5.shape).cuda()
        random_2 = generate_random_param(self.word_embeddings2.shape).cuda()
        random_3 = generate_random_param(self.word_embeddings3.shape).cuda()
        random_4 = generate_random_param(self.word_embeddings4.shape).cuda()
        random_7 = generate_random_param(self.word_embedding_gpt2.shape).cuda()
        random_8 = generate_random_param(self.word_embedding_qwen.shape).cuda()


        self.model_embedding_list=[self.word_embeddings5,self.word_embeddings2,self.word_embeddings3,self.word_embeddings4,self.word_embedding_gpt2,self.word_embedding_qwen]
        self.num_tokens = 1000
        self.mapping = MLPS_for_reprogram(len(self.word_embeddings2))


        tokenizer = [self.tokenizer5,self.tokenizer2,self.tokenizer3,self.tokenizer4,self.tokenizer_gpt2,self.tokenizer6]
        llm = [self.llm_model4,self.llm_model2,self.llm_model3,self.llm_model4,self.llm_modelgpt,self.llm_model6]
        self.MoE = MoE(
            64, 4, llm, tokenizer=tokenizer,
            use_shared_expert=True,
            shared_group=(1,2,3,4),
            shared_lambd=0.1,
        )

        self.linear_layer = nn.Linear(30522, self.num_tokens)

        if (self.feature == 'text' or self.feature == 'id+text'):
            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()

            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 0
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")
                tensor = tensor.to(0)
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor = torch.nn.Parameter(self.bert_tensor.detach())

                self.mlps = MLPS(self.emb_size)
            elif (len(self.datasetFile.split(",")) > 1):
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.datasetFile.split(","):
                    if not os.path.exists(dataset + "whole_tensor.pt"):
                        mask = 0
                        pre = Pretrain(self.data, dataset, mask)
                    tensor = torch.load(dataset + "whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)


                self.mlps = MLPS(self.emb_size)

        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 1, self.emb_size)))

        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len + 1, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-12)


    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def forward(self, seq, pos):

        seq = torch.tensor(seq, device=current_device)
        pos = torch.tensor(pos, device=current_device)
        if (self.feature == 'text'):


            seq_emb = self.bert_tensor[seq]
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq])
        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = (seq == 0).to(dtype=torch.bool, device=current_device)

        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=current_device))

        seq_emb_moe = self.MoE(seq_emb, seq,self.model_embedding_list)
        seq_emb = self.item_encoder(seq_emb, attention_mask, seq_emb_moe, timeline_mask,
                                    output_all_encoded_layers=False)

        seq_emb = seq_emb[-1]

        return seq_emb

    def init_weights(self, module):
\

        if isinstance(module, (nn.Linear, nn.Embedding)):


            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def gelu(x):
\
\
\
\
\
\

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
\

        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()

        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.H),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):

        logits = self.classifier(bert_tensor)


        return logits


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=1024, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, 64)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class MoE(nn.Module):
    def __init__\
                    (self,
                     input_dim,
                     output_dim,
                     llm,
                     tokenizer,
                     num_experts=6, dropout_rate=0.1, noisy_gating=False,
                     noise_epsilon=1e-2,
                     use_shared_expert: bool = False,
                     shared_group=(1, 2, 3),
                     shared_lambd: float = 1.0,
                     ):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon


        dimension_mapping_size = [4096, 768, 768, 768,768,1024]

        assert num_experts == len(dimension_mapping_size), "num_experts 必须和 dimension_mapping_size 长度一致"

        self.experts = nn.ModuleList(
            [


                ReprogrammingLayer(input_dim, output_dim, d_llm=in_dim)
                for in_dim in dimension_mapping_size
            ]
        )
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_experts)
        ])


        self.gating_network = GatingNetwork(input_dim, num_experts)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts))


        mapping_sizes = [32000, 30522, 32100, 32100, 50257, 151936]

        dimension_mapping_size=[4096,768,768,768,768,1024]

        self.dimension_mappings = nn.ModuleList([
            nn.Linear(size, 64) for size in dimension_mapping_size[:num_experts]
        ])
        self.mappings = nn.ModuleList([
            MLPS_for_reprogram(size) for size in mapping_sizes[:num_experts]
        ])
        txt_file_path ='/root/autodl-tmp/dataset/Amazon-Music/gppt4_200_products'


        with open(txt_file_path, 'r') as file:
            word_list = [line.strip() for line in file if line.strip()]
        self.word_embeddinglist = []
        self.word_embeddinglist_copy = []
        all_words_as_sentence = " ".join(word_list)
        for i in range(6):


            inputs = tokenizer[i](all_words_as_sentence, return_tensors="pt", padding=False, truncation=True).to('cuda')
            input_ids = inputs["input_ids"].cuda()


            embeddings = llm[i].get_input_embeddings().cuda()(input_ids).detach().clone().requires_grad_(
                True)


            embeddings = embeddings.squeeze(0)

            embedding_copy = torch.rand_like(embeddings)
            self.word_embeddinglist.append(embeddings)
            self.word_embeddinglist_copy.append(embedding_copy)
        self.use_shared_expert = use_shared_expert
        self.shared_group = tuple(shared_group)
        self.shared_lambda = float(shared_lambd)


        self.shared_expert = None
        if self.use_shared_expert:
            self.shared_expert = copy.deepcopy(self.experts[self.shared_group[0]])


            for p in self.shared_expert.parameters():
                p.requires_grad = False


    def forward(self, seq_emb, seq, word_embeddings_list):

        clean_logits = self.gating_network(seq_emb)


        if self.noisy_gating:
            raw_noise_stddev = seq_emb @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits).to(seq_emb.device) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        expert_weights = F.softmax(logits, dim=-1).permute(0, 2, 1).unsqueeze(-1)
        cached_sources=[]
        expert_outputs = []
        for i in range(self.num_experts):

            source_embeddings= self.process_and_select_grad(seq_emb,word_embeddings_list[i],self.dimension_mappings[i], seq)

            output = self.dropouts[i](self.experts[i](seq_emb, source_embeddings, source_embeddings))
            if self.use_shared_expert and (self.shared_expert is not None) and (i in self.shared_group):
                cached_sources.append(source_embeddings)
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(expert_outputs * expert_weights, dim=1)


        if self.use_shared_expert and (self.shared_expert is not None):

            if len(cached_sources) > 0:

                try:
                    shared_source = torch.stack(cached_sources, dim=0).mean(dim=0)
                except Exception:
                    shared_source = cached_sources[0]
            else:
                shared_source = None

            if shared_source is not None:
                shared_out = self.shared_expert(seq_emb, shared_source, shared_source)
                output = output + self.shared_lambda * shared_out

        return output

    @torch.no_grad()
    def build_shared_from_round1(self, round1_pack: dict, device=None):
\
\
\
\

        assert self.shared_expert is not None, "shared_expert not enabled. Set use_shared_expert=True."
        fisher_list = round1_pack["fisher_experts"]

        if "moe_experts_state_dict" in round1_pack:
            experts_sd = round1_pack["moe_experts_state_dict"]
        else:
            raise KeyError("round1_pack must contain `moe_experts_state_dict` (recommended).")


        tmp_experts = copy.deepcopy(self.experts)
        tmp_experts.load_state_dict(experts_sd, strict=True)

        if device is None:
            device = next(self.parameters()).device


        merged_sd = {}
        group = self.shared_group


        for name, p_shared in self.shared_expert.named_parameters():
            num = torch.zeros_like(p_shared, device=device)
            den = torch.zeros_like(p_shared, device=device)

            for eid in group:

                p_i = dict(tmp_experts[eid].named_parameters())[name].to(device)
                F_i = fisher_list[eid][name].to(device)

                num += F_i * p_i
                den += F_i

            merged_sd[name] = num / (den + 1e-12)


        for name, p in self.shared_expert.named_parameters():
            p.copy_(merged_sd[name].to(p.device))


        for p in self.shared_expert.parameters():
            p.requires_grad_(False)

        print(f"[SharedExpert] built from round1 fisher+weights, group={group}, frozen=True")
    def process_and_select_grad(self, A: torch.Tensor, B: torch.Tensor, linear_mapping, seq):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


        Bsz, L, d_low = A.shape
        V, D_high = B.shape


        seq_mask = (seq != 0).unsqueeze(-1).float()
        A_sum = (A * seq_mask).sum(dim=1)
        cnt = seq_mask.sum(dim=1).clamp(min=1.0)
        A_avg = A_sum / cnt


        B_high = B
        linear_mapping = linear_mapping.to(B.device)
        B_low = linear_mapping(B_high)


        A_norm = F.normalize(A_avg, dim=-1)
        B_norm = F.normalize(B_low, dim=-1)
        sim = A_norm @ B_norm.T


        K = 10
        tau = 0.5
        eps = 1e-10


        g = -torch.log(-torch.log(torch.rand_like(sim) + eps) + eps)
        logits = (sim + g) / tau
        y_soft = F.softmax(logits, dim=-1)


        _, topk_idx = torch.topk(y_soft, K, dim=-1)
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(1, topk_idx, 1.0)


        y_st = (y_hard - y_soft).detach() + y_soft


        selected_tokens = B_high[topk_idx]


        selected_w = y_st.gather(1, topk_idx)


        selected_w = selected_w / (selected_w.sum(dim=-1, keepdim=True) + 1e-8)


        selected_tokens = selected_tokens * selected_w.unsqueeze(-1)


        selected_tokens_flat = selected_tokens.reshape(Bsz * K, D_high)


        return selected_tokens_flat
    def encode_and_expand(self, word_embedding, i):

        embeddings = self.word_embeddinglist[i]

        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        word_embedding_normalized = F.normalize(word_embedding, p=2, dim=1)


        similarities = torch.matmul(embeddings_normalized, word_embedding_normalized.T)
        nearest_indices = torch.topk(similarities, 5, dim=1, largest=True).indices.squeeze().to(
            word_embedding.device)
        nearest_indices = nearest_indices.reshape(-1)

        expanded_embeddings = torch.cat((embeddings, word_embedding[nearest_indices]),
                                        dim=0)

        return expanded_embeddings


class MLPS_for_reprogram(nn.Module):
    def __init__(self, H):
        super(MLPS_for_reprogram, self).__init__()

        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(self.H, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1000),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):

        logits = self.classifier(bert_tensor)
        return logits


class BSARecEncoder(nn.Module):
    def __init__(self,blocknum):
        super(BSARecEncoder, self).__init__()

        block = BSARecBlock()
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(blocknum)])

    def forward(self, hidden_states, attention_mask, seq_emb_moe, timeline_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for idx,layer_module in enumerate(self.blocks):
            hidden_states = layer_module(hidden_states, attention_mask)
            if idx==0:
                hidden_states +=seq_emb_moe
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BSARecBlock(nn.Module):
    def __init__(self):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer()
        self.feed_forward = FeedForward()

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        hidden_size = 64
        inner_size = 4 * hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act("gelu")

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.5)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
\
\
\
\
\
\
\

        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class BSARecLayer(nn.Module):
    def __init__(self):
        super(BSARecLayer, self).__init__()

        self.filter_layer = FrequencyLayer()
        self.attention_layer =nn.MultiheadAttention(
    embed_dim=64,
    num_heads=1,
    batch_first=True
)
        self.alpha = 0.9

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp, _ = self.attention_layer(
            query=input_tensor,
            key=input_tensor,
            value=input_tensor,
            attn_mask=attention_mask
        )
        hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp

        return hidden_states


class FrequencyLayer(nn.Module):
    def __init__(self):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(0.5)
        self.LayerNorm = LayerNorm(64, eps=1e-12)
        self.c = 3 // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, 64))

    def forward(self, input_tensor):

        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class MLPS_for_reprogram2(nn.Module):
    def __init__(self):
        super(MLPS_for_reprogram2, self).__init__()


        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):

        logits = self.classifier(bert_tensor)


        return logits
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()


        num_attention_heads=1
        hidden_size=64
        self.num_attention_heads =num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(0.5)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(0.5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)


        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size


        attention_scores = attention_scores + attention_mask


        attention_probs = self.softmax(attention_scores)


        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
