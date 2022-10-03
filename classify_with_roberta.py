import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)

import math
from transformers import  (
    BertPreTrainedModel,
    RobertaConfig,
    RobertaTokenizerFast
)

from transformers.optimization import (
    AdamW,
    get_linear_schedule_with_warmup
)

from scipy.special import softmax
from torch.nn import CrossEntropyLoss

from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
    average_precision_score,
)

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)

segments = pd.read_csv("data_n100_labelled.csv")[['text','crime']]
train_df = segments.iloc[0:80, :]
test_df = segments.iloc[81:100, :]

model_name = 'roberta-base'
num_labels = 2
device = torch.device("cpu")

tokenizer_name = model_name

max_seq_length = 512
train_batch_size = 8
test_batch_size = 8
warmup_ratio = 0.06
weight_decay=0.0
gradient_accumulation_steps = 1
num_train_epochs = 25
learning_rate = 1e-05
adam_epsilon = 1e-08

class RobertaForSmilesClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForSmilesClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)


    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

config_class = RobertaConfig
model_class = RobertaForSmilesClassification
tokenizer_class = RobertaTokenizerFast

config = config_class.from_pretrained(model_name, num_labels=num_labels)

model = model_class.from_pretrained(model_name, config=config)

tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)


class MyClassificationDataset(Dataset):

    def __init__(self, data, tokenizer):
        text, labels = data
        self.examples = tokenizer(text=text,text_pair=None,truncation=True,padding="max_length",
                                  max_length=max_seq_length,return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)


    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = MyClassificationDataset(train_examples,tokenizer)

test_examples = (test_df.iloc[:, 0].astype(str).tolist(), test_df.iloc[:, 1].tolist())
test_dataset = MyClassificationDataset(test_examples,tokenizer)

def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(device) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(device)
    return inputs

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=train_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

#Extract a batch as sanity-check
batch = get_inputs_dict(next(iter(train_dataloader)))
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device)

class MyClassificationDataset(Dataset):

    def __init__(self, data, tokenizer):
        text, labels = data
        self.examples = tokenizer(text=text,text_pair=None,truncation=True,padding="max_length",
                                  max_length=max_seq_length,return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)


    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = MyClassificationDataset(train_examples,tokenizer)

test_examples = (test_df.iloc[:, 0].astype(str).tolist(), test_df.iloc[:, 1].tolist())
test_dataset = MyClassificationDataset(test_examples,tokenizer)

def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(device) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(device)
    return inputs

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=train_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

#Extract a batch as sanity-check
batch = get_inputs_dict(next(iter(train_dataloader)))
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device)


t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
optimizer_grouped_parameters = []
custom_parameter_names = set()
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters.extend(
    [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
)

warmup_steps = math.ceil(t_total * warmup_ratio)
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

def compute_metrics(preds, model_outputs, labels, eval_examples=None, multi_label=False):
    assert len(preds) == len(labels)
    mismatched = labels != preds
    wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    scores = np.array([softmax(element)[1] for element in model_outputs])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    auprc = average_precision_score(labels, scores)
    return (
        {
            **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "auroc": auroc, "auprc": auprc},
        },
        wrong,
    )

def print_confusion_matrix(result):
    print('confusion matrix:')
    print('            predicted    ')
    print('          0     |     1')
    print('    ----------------------')
    print('   0 | ',format(result['tn'],'5d'),' | ',format(result['fp'],'5d'))
    print('gt -----------------------')
    print('   1 | ',format(result['fn'],'5d'),' | ',format(result['tp'],'5d'))
    print('---------------------------------------------------')


model.to(device)

model.zero_grad()

for epoch in range(num_train_epochs):

    model.train()
    epoch_loss = []

    for batch in train_dataloader:
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        epoch_loss.append(loss.item())

    #evaluate model with test_df at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_dataloader)
    preds = np.empty((len(test_dataset), num_labels))
    out_label_ids = np.empty((len(test_dataset)))
    model.eval()

    for i,test_batch in enumerate(test_dataloader):
        with torch.no_grad():
            test_batch = get_inputs_dict(test_batch)
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            labels = test_batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(test_dataset)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    result, wrong = compute_metrics(preds, model_outputs, out_label_ids, test_examples)

    print('epoch',epoch,'Training avg loss',np.mean(epoch_loss))
    print('epoch',epoch,'Testing  avg loss',eval_loss)
    print(result)
    print_confusion_matrix(result)
    print('---------------------------------------------------\n')

torch.save(model.state_dict(), "./roberta.pt")
