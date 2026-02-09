# MLTFR

MLTFR is a plug-and-play framework leveraing LLM knowledge to do sequential recommendation.

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
BERT4Rec=-n_blocks 2 -drop_rate 0.2 -n_heads 1 -mask_rate 0.5 -eps 0.1 -lambda 0.001 -cl 0 -cltype text -strategy 1    #strategy1:Direct Learning Strategy    strategy2: User Information Strategy strategy3: External
Knowledge Strategy
output.setup=-dir ./results/
feature=id     #default feature=id
```

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








