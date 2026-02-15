# fisher_expert_utils.py
import os
import torch
import torch.nn.functional as F
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.loss_torch import l2_reg_loss,InfoNCE,batch_softmax_loss
from data.augmentor import SequenceAugmentor
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
import random
local_random = random.Random(2024)
import torch


#计算、保存和加载fisher矩阵

def load_round1_and_enable_shared(model, round1_ckpt_path, enable_shared=True):
    pack = torch.load(round1_ckpt_path, map_location="cpu")

    # 1) load round1 full model
    if "full_model_state_dict" in pack:
        model.load_state_dict(pack["full_model_state_dict"], strict=False)
        print("[Round2] loaded full_model_state_dict from round1")
    else:
        print("[Round2] WARNING: no full_model_state_dict found, skip loading backbone.")

    # 2) enable shared expert
    if enable_shared:
        model.MoE.use_shared_expert = True

        # shared_expert must exist
        if getattr(model.MoE, "shared_expert", None) is None:
            import copy
            model.MoE.shared_expert = copy.deepcopy(model.MoE.experts[model.MoE.shared_group[0]])
            for p in model.MoE.shared_expert.parameters():
                p.requires_grad_(False)
        p0 = next(model.MoE.shared_expert.parameters())
        # 3) build shared expert from fisher + experts weights
        print("before:", p0.data.norm().item())
        model.MoE.build_shared_from_round1(pack)
        print("after:", p0.data.norm().item())
    return pack

def _init_fisher_for_experts(moe_module: torch.nn.Module):
    """
    fisher[i][name] = zeros_like(param)
    """
    fisher = []
    for exp in moe_module.experts:
        exp_f = {}
        for name, p in exp.named_parameters():
            exp_f[name] = torch.zeros_like(p, device=p.device)
        fisher.append(exp_f)
    return fisher


def _accumulate_fisher(moe_module: torch.nn.Module, fisher_buf):
    """
    fisher += grad^2
    """
    for i, exp in enumerate(moe_module.experts):
        for name, p in exp.named_parameters():
            if p.grad is None:
                continue
            fisher_buf[i][name] += (p.grad.detach() ** 2)


@torch.no_grad()
def _finalize_fisher(fisher_buf, denom: int, to_cpu: bool = True):
    denom = max(int(denom), 1)
    for i in range(len(fisher_buf)):
        for k in fisher_buf[i]:
            fisher_buf[i][k] /= denom
            if to_cpu:
                fisher_buf[i][k] = fisher_buf[i][k].cpu()
    return fisher_buf


def compute_fisher_experts_from_bert4rec(
    bert4rec_obj,
    model: torch.nn.Module,
    epoch: int,
    save_path: str,
    model_name: str,
    max_batches: int = None,
    use_eval_mode: bool = True,
    include_cl_loss: bool = True,
    disable_noisy_gating: bool = True,

):

    device = next(model.parameters()).device

    # ---- locate MoE ----
    if not hasattr(model, "MoE"):
        raise AttributeError("Your BERT_Encoder model does not have attribute `MoE`.")
    moe = model.MoE

    # ---- mode setting ----
    old_train_state = model.training
    if use_eval_mode:
        model.eval()
    else:
        model.train()

    # ---- optionally disable noisy gating ----
    old_noisy = None
    if disable_noisy_gating and hasattr(moe, "noisy_gating"):
        old_noisy = moe.noisy_gating
        moe.noisy_gating = False

    fisher_buf = _init_fisher_for_experts(moe)

    # ---- iterate batches like your training loop ----
    n = 0
    data = bert4rec_obj.data
    bs = bert4rec_obj.batch_size
    labelgap = bert4rec_obj.labelgap
    max_len = bert4rec_obj.max_len

    for bidx, batch in enumerate(next_batch_sequence(data, bs, labelgap, max_len=max_len)):
        if max_batches is not None and bidx >= max_batches:
            break

        seq, seqfull, pos, posfull, y, neg_idx, seq_len, labelgap = batch

        # ============ 1) main forward ============
        if model_name=='BSARec' or model_name=='SASRec' or model_name=='FMLPRec':

            seq_emb = model.forward(seq, pos)  # [B, L, H]
            rec_loss = bert4rec_obj.calculate_loss(seq_emb, y, neg_idx, pos)
            batch_loss = rec_loss
        # ============ 2) CL augmentations ============

        if model_name=='CL4SRec':

            switch = local_random.sample(range(3), k=2)
            aug_type = switch[0]

            # ---- aug view 1 ----
            if aug_type == 0:
                aug_seq1, aug_pos1, aug_len1 = SequenceAugmentor.item_crop(seq, seq_len, bert4rec_obj.aug_rate)
                aug_emb1 = model.forward(aug_seq1, aug_pos1)
                cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, bert4rec_obj.emb_size) for i, last in enumerate(aug_len1)]
            elif aug_type == 1:
                aug_seq1 = SequenceAugmentor.item_reorder(seq, seq_len, bert4rec_obj.aug_rate)
                aug_emb1 = model.forward(aug_seq1, pos)
                cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, bert4rec_obj.emb_size) for i, last in enumerate(seq_len)]
            elif aug_type == 2:
                aug_seq1 = SequenceAugmentor.item_mask(seq, seq_len, bert4rec_obj.aug_rate, bert4rec_obj.data.item_num + 1)
                aug_emb1 = model.forward(aug_seq1, pos)
                cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, bert4rec_obj.emb_size) for i, last in enumerate(seq_len)]
            else:
                raise ValueError(f"Unknown aug_type={aug_type}")

            # ---- aug view 2 ----
            if aug_type == 0:
                aug_seq2, aug_pos2, aug_len2 = SequenceAugmentor.item_crop(seq, seq_len, bert4rec_obj.aug_rate)
                aug_emb2 = model.forward(aug_seq2, aug_pos2)
                cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, bert4rec_obj.emb_size) for i, last in enumerate(aug_len2)]
            elif aug_type == 1:
                aug_seq2 = SequenceAugmentor.item_reorder(seq, seq_len, bert4rec_obj.aug_rate)
                aug_emb2 = model.forward(aug_seq2, pos)
                cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, bert4rec_obj.emb_size) for i, last in enumerate(seq_len)]
            elif aug_type == 2:
                aug_seq2 = SequenceAugmentor.item_mask(seq, seq_len, bert4rec_obj.aug_rate, bert4rec_obj.data.item_num + 1)
                aug_emb2 = model.forward(aug_seq2, pos)
                cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, bert4rec_obj.emb_size) for i, last in enumerate(seq_len)]


            cl_loss = bert4rec_obj.cl_rate * InfoNCE(
                torch.cat(cl_emb1, dim=0),
                torch.cat(cl_emb2, dim=0),
                1,
                True
            )


            rec_loss = bert4rec_obj.calculate_loss(seq_emb, y, neg_idx, pos)

            # l2 reg
            reg_loss = l2_reg_loss(bert4rec_obj.reg, model.item_emb)

            batch_loss = rec_loss + reg_loss + cl_loss
        if model_name == 'BERT4Rec' or model_name == 'BERTLin':

            # 3)掩码loss
            aug_seq, masked, labels = bert4rec_obj.item_mask_for_bert(seq, seq_len, bert4rec_obj.aug_rate,
                                                             bert4rec_obj.data.item_num + 1)
            seq_emb = model.forward(aug_seq, pos)
            batch_loss = bert4rec_obj.calculate_loss(seq_emb, masked, labels)

        # ---- backward for Fisher ----
        model.zero_grad(set_to_none=True)
        batch_loss.backward()

        # accumulate only experts[i] Fisher
        _accumulate_fisher(moe, fisher_buf)

        n += 1

    # ---- finalize fisher ----
    fisher_buf = _finalize_fisher(fisher_buf, denom=n, to_cpu=True)
    full_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    experts_state = moe.experts.state_dict()
    experts_state = {k: v.detach().cpu() for k, v in experts_state.items()}

    pack = {
        "epoch": int(epoch),
        "num_batches": int(n),
        "fisher_experts": fisher_buf,

        # ---- NEW ----
        # "full_model_state_dict": full_model_state,
        "moe_experts_state_dict": experts_state,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    torch.save(pack, save_path)

    # ---- restore states ----
    if disable_noisy_gating and (old_noisy is not None):
        moe.noisy_gating = old_noisy
    model.train(old_train_state)

    print(f"[FISHER] saved experts Fisher to: {save_path} (batches={n})")
    return pack
