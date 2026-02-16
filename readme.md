# MLTFR

MLTFR is a plug-and-play framework that leverages LLM knowledge for sequential recommendation. It supports flexible selection of LLM vocabulary parameters, which are filtered and then fused into a sequential recommender without requiring any additional textual corpora. MLTFR is trained in two stages: the first stage estimates the Fisher information matrix, which serves as evidence for integrating shared experts in the second stage.
The steps to run the code are as follows:

**Step1：** Set the model hyperparameters you want to run in MMKEI/conf/model.conf, an example is:

```
training.set=./dataset/Amazon-Pantry/train.txt   #training path
test.set=./dataset/Amazon-Pantry/test.txt  #test path
dataset=./dataset/Amazon-Pantry/
model.name=BERT4Rec
model.type=sequential
item.ranking=-topN 10,20
embedding.size=64
num.max.epoch=100
batch_size=128
learnRate=0.001
reg.lambda=0.0001
max_len=50
BERT4Rec=-n_blocks 2 -drop_rate 0.2 -n_heads 1 -mask_rate 0.5 -eps 0.1 -lambda 0.001 -cl 0 -cltype text -strategy 2  -round2_enable_shared 1   #strategy2: Our User Information Strategy 
output.setup=-dir ./results/
feature=id     #default feature=id
```


round2_enable_shared indicates whether to load the shared experts. In Round 1, we set round2_enable_shared to 0 and enable the Fisher computation code in MMKEI/Model/Pre as follows:
```
save_path = f"./fisher_cache/fisher_experts_epoch_BSA_test{self.maxEpoch}.pt"
        compute_fisher_experts_from_bert4rec(
            bert4rec_obj=self,
            model=model,
            epoch=self.maxEpoch,
            save_path=save_path,
            model_name=self.model_name,
            max_batches=None,
            use_eval_mode=True,
            include_cl_loss=True,
            disable_noisy_gating=True,
        )
```
In Round 2, we set round2_enable_shared to 1. Meanwhile, in the MoE module, use_shared_expert=True is enabled by default, which fuses a group of experts with the same architecture.

**Step2：** run the code:
```python main.py```
choose the backbone model:

```
MMKEI: A library for Sequential Recommendation. 
================================================================================
Baseline Models:
SASRec   BERT4Rec   CL4SRec   UnisRec   FMLPRec   LinRec   STRec   BERTLin
--------------------------------------------------------------------------------
Please enter the baselines you want to run:BERT4Rec
stages:
Pre   Fine
--------------------------------------------------------------------------------
Please enter the stage you want:Pre    #MMKEI has only one stage, so it is written as Pre.
```

**Tips:**
1.To download the large language model, obtain the vocabulary, or use your own large language model , you need to replace the code in 'MMKEI/model/module/modelname' with the path of your own model. All models can be downloaded from Hugging Face.

```
self.tokenizer = T5TokenizerFast.from_pretrained(
            'P5-beautybase',legacy=False)
 self.llm_model = T5Model.from_pretrained(
            'P5-beautybase',
            local_files_only=True,  ).cuda()
```








