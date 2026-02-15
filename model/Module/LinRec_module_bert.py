import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig,\
    BertModel, BertTokenizer,XLMRobertaTokenizer, XLMRobertaModel,T5Tokenizer, T5Model,T5TokenizerFast
from torch.nn.functional import normalize
from data.pretrain import Pretrain
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from math import sqrt
import math
import copy

torch.cuda.set_device(0)
current_device = torch.cuda.current_device()


class LinRec_Model_bert(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile,strategy):
        super(LinRec_Model_bert, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.datasetFile = datasetFile
        self.strategy = strategy
        self._init_model()
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
        self.word_embeddings4 = self.llm_model4.get_input_embeddings().weight
        self.word_embeddings5 = self.llama.get_input_embeddings().weight
        self.word_embedding_gpt2 = self.llm_modelgpt.get_input_embeddings().weight

        self.word_embeddings6 =self.llm_model6.get_input_embeddings().weight
        self.word_embeddings7 = self.word_embeddings2.clone()
        self.word_embeddings8 = self.word_embeddings2.clone()
        self.word_embeddings9 = self.word_embeddings2.clone()
        self.word_embeddings10 = self.word_embeddings2.clone()
        self.word_embeddings3_1 = self.word_embeddings3.clone()
        self.word_embeddings4_1 = self.word_embeddings4.clone()
        self.word_embedding_gpt2_1 = self.word_embedding_gpt2.clone()
        self.word_embedding_gpt2_2 = self.word_embedding_gpt2.clone()

        def generate_random_param(shape):
            return nn.Parameter(torch.normal(mean=0.0, std=0.02, size=shape))


        random_5 = generate_random_param(self.word_embeddings5.shape).cuda()
        random_2 = generate_random_param(self.word_embeddings2.shape).cuda()
        random_3 = generate_random_param(self.word_embeddings3.shape).cuda()
        random_4 = generate_random_param(self.word_embeddings4.shape).cuda()
        random_7 = generate_random_param(self.word_embedding_gpt2.shape).cuda()
        random_8 = generate_random_param(self.word_embeddings6.shape).cuda()


        self.model_embedding_list=[self.word_embeddings5 ,self.word_embeddings2 ,self.word_embeddings3 ,self.word_embeddings4]
        self.num_tokens = 1000
        self.mapping = MLPS_for_reprogram(len(self.word_embeddings2))
        self.reprogramming_layer = ReprogrammingLayer(64, 8, d_llm=768, d_keys=32)

        tokenizer = [self.tokenizer5,self.tokenizer2,self.tokenizer3,self.tokenizer4]
        llm = [self.llama ,self.llm_model2,self.llm_model3,self.llm_model4]
        self.MoE = MoE(
            64, 4, llm, tokenizer=tokenizer,
            use_shared_expert=True
            ,
            shared_group=(1,2,3),
            shared_lambd=0.1,
        )

        self.linear_layer = nn.Linear(30522, self.num_tokens)
    def _init_model(self):
        initializer = nn.init.xavier_uniform_

        if (self.feature == 'text' or self.feature == 'id+text'):
            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()

            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 0
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")
                tensor = tensor.to(1)
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)

                self.mlps = MLPS(768)
            elif (len(self.datasetFile.split(",")) > 1):
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.datasetFile.split(","):
                    if not os.path.exists(dataset + "whole_tensor.pt"):
                        mask = 0
                        pre = Pretrain(self.data, dataset, mask)
                    tensor = torch.load(dataset + "whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)


                self.mlps = MLPS(self.emb_size)

        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 2, self.emb_size)))

        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len + 2, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer = MultiHeadAttention(self.head_num, self.emb_size, 0.2, 0.2, 1e-12)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))

            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)


    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, seq, pos):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        if (self.feature == 'text'):
            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq.cuda()])
        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()

        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]


        for i in range(len(self.attention_layers)):
            if (i == 0):
                seq_emb_moe = self.MoE(seq_emb, seq.cuda(),self.strategy,
                                       self.model_embedding_list)
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb= self.attention_layers[i](seq_emb)

            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
            if (i == 0):
                seq_emb = seq_emb + seq_emb_moe
        seq_emb = self.last_layer_norm(seq_emb)
        seq_emb_input = seq_emb

        return seq_emb


class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()

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
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):

        logits = self.classifier(bert_tensor)


        return logits


class ReprogrammingLayerLin(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=1024, attention_dropout=0.1, layer_norm_eps=1e-12):
        super(ReprogrammingLayerLin, self).__init__()


        d_keys = d_keys or (d_model // n_heads)
        self.attention_head_size = d_keys
        self.all_head_size = d_keys * n_heads


        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)


        self.out_projection = nn.Linear(self.all_head_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(attention_dropout)

        self.n_heads = n_heads
        self.elu = nn.ELU()
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.n_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads


        query_layer = self.query_projection(target_embedding).view(B, L, H, -1)
        key_layer = self.key_projection(source_embedding).view(S, H, -1)
        value_layer = self.value_projection(value_embedding).view(S, H, -1)


        elu_query = self.elu(query_layer)
        elu_key = self.elu(key_layer)


        query_norm_inverse = 1 / torch.norm(elu_query, dim=3, p=2, keepdim=True)
        key_norm_inverse = 1 / torch.norm(elu_key, dim=2, p=2, keepdim=True)


        normalized_query_layer = elu_query * query_norm_inverse
        normalized_key_layer = elu_key * key_norm_inverse


        scores = torch.einsum("blhe,she->bhls", normalized_query_layer,normalized_key_layer )

        scale = 1. / sqrt(self.attention_head_size)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        context_layer = torch.einsum("bhls,she->blhe", A, value_layer)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.out_projection(context_layer)


        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + target_embedding)

        return hidden_states
class MultiHeadAttention(nn.Module):
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


    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax_col = nn.Softmax(dim=-2)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor):

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)


        elu = nn.ELU()

        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        query_norm_inverse = 1 / torch.norm(elu_query, dim=3, p=2)
        key_norm_inverse = 1 / torch.norm(elu_key, dim=2, p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij', elu_query, query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij', elu_key, key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer, torch.matmul(normalized_key_layer,
                                                                          value_layer)) / self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


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
    def __init__(self, input_dim, output_dim, llm,tokenizer,num_experts=4, dropout_rate=0.1, noisy_gating=False,
                 noise_epsilon=1e-2,use_shared_expert: bool = False,
                     shared_group=(1, 2, 3),
                     shared_lambd: float = 1.0,):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon

        dimension_mapping_size = [4096,768,768,768]
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


        mapping_sizes = [32000,30522,32100,32100]

        dimension_mapping_size=[4096,768,768,768]
        self.dimension_mappings= nn.ModuleList([
           nn.Linear(size,64) for size in dimension_mapping_size[:num_experts]
        ])
        self.mappings = nn.ModuleList([
            MLPS_for_reprogram(size) for size in mapping_sizes[:num_experts]
        ])
        txt_file_path = '/root/autodl-tmp/dataset/Amazon-Music/gpt4_200_key'

        with open(txt_file_path, 'r') as file:
            word_list = [line.strip() for line in file if line.strip()]
        self.word_embeddinglist = []
        self.word_embeddinglist_copy=[]
        all_words_as_sentence = " ".join(word_list)
        for i in range(4):


            inputs = tokenizer[i](all_words_as_sentence, return_tensors="pt", padding=False, truncation=True).to('cuda')
            input_ids = inputs["input_ids"].cuda()


            embeddings = llm[i].get_input_embeddings().cuda()(input_ids).detach().clone().requires_grad_(
                True)


            embeddings = embeddings.squeeze(0)

            embedding_copy=torch.rand_like(embeddings)
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


    def forward(self, seq_emb,seq,strategy,word_embeddings_list):

        clean_logits = self.gating_network(seq_emb)


        if self.noisy_gating:
            raw_noise_stddev = seq_emb @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits).to(seq_emb.device) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        expert_weights = F.softmax(logits, dim=-1).permute(0, 2, 1).unsqueeze(-1)

        expert_outputs = []
        cached_sources=[]
        for i in range(self.num_experts):
            if strategy==1:
              source_embeddings = self.mappings[i](word_embeddings_list[i].permute(1, 0)).permute(1, 0)
            if strategy==2:
              source_embeddings = self.process_and_select_grad(seq_emb,word_embeddings_list[i],self.dimension_mappings[i], seq)
            if strategy==3:
              source_embeddings = self.encode_and_expand(word_embeddings_list[i], i)
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


        return output

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


        K = 1
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
        if self.training and torch.is_grad_enabled():
            sim.retain_grad()
            y_soft.retain_grad()
            y_st.retain_grad()
            selected_w.retain_grad()

            self._dbg = {
                "sim": sim,
                "y_soft": y_soft,
                "y_st": y_st,
                "topk_idx": topk_idx.detach(),
                "selected_w": selected_w,
                "A_avg": A_avg,
                "B_low": B_low,
            }

        return selected_tokens_flat
    def encode_and_expand(self, word_embedding, i):
        embeddings = self.word_embeddinglist[i]

        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        word_embedding_normalized = F.normalize(word_embedding, p=2, dim=1)


        similarities = torch.matmul(embeddings_normalized, word_embedding_normalized.T)
        nearest_indices = torch.topk(similarities, 1, dim=1, largest=True).indices.squeeze().to(
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
