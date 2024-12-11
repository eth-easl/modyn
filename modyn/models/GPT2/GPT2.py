#plimport pytorch_lightning as pl

from transformers import (
    Adafactor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

import torch
#from Datasets import CustomDataset, Pretrain_Chunks
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from collections import Counter
from typing import Optional, Tuple, Dict, Union, Any, Callable
from typing import Any, Union,List
import re
import string
#from deepspeed.runtime.lr_schedules import WarmupDecayLR
#import deepspeed
import math
import os
import csv
from modyn.models.GPT2.GPT2_Model_LoRA import GPT2LMHeadModel #as GPT2_Lora
#from modyn.models.GPT2.RecAdam import RecAdam
from modyn.models.coreset_methods_support import CoresetSupportingModule


class GPT2:
    # pylint: disable-next=unused-argument
    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        self.model = GPT2Modyn(hparams)
        self.model.to(device)


# the following class is adapted from
# torchvision https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

class GPT2Modyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super(GPT2Modyn, self).__init__()
        # self.save_hyperparameters(hparams)
        self.unchanged_loss: float = 0.0
        self.changed_loss: float = 0.0
        self.invariant_loss: float = 0.0
        self.unchanged: int = 0
        self.changed: int = 0
        self.invariant: int = 0
        self.validation: int = 0
        self.validation_loss: float = 0.0
        
        self.mix_ratio: float = 1.0
        self.mix_decay: float = 0.7
        self.epoch: int = 0

        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")  # hparams.model_name_or_path

    def freeze_params(self, model: torch.nn.Module) -> None:
        for par in model.parameters():
            par.requires_grad = False

    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        def rid_of_specials(text: str) -> str:
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction: str, ground_truth: str) -> int:
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def calculate_scores(self, predictions: list[str], ground_truths: list[str]) -> tuple[float, float]:
        em_score: float = 0.0
        f1_score: float = 0.0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score += self.exact_match_score(prediction, ground_truth)
            f1_score += self._f1_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        f1_score /= len(predictions)
        return em_score * 100, f1_score * 100

    def lmap(self, f: Callable, x: list[Any]) -> list[Any]:
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self) -> bool:
        return self.trainer.global_rank <= 0

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                lm_labels: Optional[torch.Tensor] = None) -> Any:
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

    def get_last_layer(self) -> torch.nn.Module:
        return self.model.lm_head

    """
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
        )

        loss = outputs[0]
        return loss

    def valid_step(self, batch):
        lm_labels = batch["label_ids"].clone().detach()
        lm_labels[source_nonprompt_mask == 0] = -100
        outputs = self(
            input_ids=batch["label_ids"],
            attention_mask=batch["label_mask"],
            lm_labels=lm_labels,
        )

        loss = outputs[0]
        print(loss)
        return loss
    
    

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
    def _generative_step_finetune(self, batch, batch_idx):
        loss = self.valid_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation +=1
        self.validation_loss += loss
        average_loss = self.validation_loss / self.validation 
        ppl = torch.exp(average_loss)
        self.log('validation_ppl', ppl, prog_bar=True, logger=True)
        
        source = self.ids_to_clean_text(batch["source_ids"])
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            max_length=self.hparams.max_input_length + 5,
            num_beams=2,
            early_stopping=True
        )
        targets = self.ids_to_clean_text(batch["target_ids"])

        generated_ids = torch.transpose(torch.transpose(generated_ids,0,1)[self.hparams.max_input_length:],0,1)
        preds = self.ids_to_clean_text(generated_ids)
        clean_preds = []
        for text in preds:
            if "." in text:
                clean_preds.append(text[:text.find(".")+1])
            else: 
                clean_preds.append(text)
        print("clean_preds",clean_preds)
        print("targets",targets)

        em_score, f1_score = self.calculate_scores(clean_preds, targets)
        print(em_score, f1_score, ppl)
        self.log('EM score', em_score, prog_bar=True, logger=True)
        self.log('F1 score', f1_score, prog_bar=True, logger=True)
    
     
    def _generative_step(self, batch, batch_idx, dataloader_idx=-1):
        loss = self._step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if dataloader_idx == 0:
            self.unchanged +=1
            self.unchanged_loss += loss
            average_loss = self.unchanged_loss / self.unchanged 
            ppl = torch.exp(average_loss)
            self.log('UnC_ppl', ppl, prog_bar=True, logger=True)
            print('UnC_ppl', ppl)
        elif dataloader_idx == 1:
            self.changed +=1
            self.changed_loss += loss
            average_loss = self.changed_loss / self.changed 
            ppl = torch.exp(average_loss)
            self.log('C_ppl', ppl, prog_bar=True, logger=True)
            print('C_ppl', ppl)
        else:
            self.invariant +=1
            self.invariant_loss += loss
            average_loss = self.invariant_loss / self.invariant 
            ppl = torch.exp(average_loss)
            self.log('IL_ppl', ppl, prog_bar=True, logger=True)
            print('IL_ppl', ppl)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def on_train_epoch_end(self):
        if self.hparams.mode=='pretrain_brute':
            self.dataset_index+=1
            if self.dataset_index==self.hparams.num_files:
                self.global_epoch+=1
                self.log('global_epoch', self.global_epoch, prog_bar=True, logger=True)
                self.dataset_index=0
            self.train_dataloader()
        if self.hparams.method=='mixreview':
            train_set = self.train_dataloader().dataset
        self.epoch+=1
     
    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        if self.hparams.mode == 'finetune':
            return self._generative_step_finetune(batch, batch_idx, dataloader_idx)
        return self._generative_step(batch, batch_idx, dataloader_idx)
    
    def configure_optimizers(self, train_len=None):
        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method=='recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 'gpt2'
            recadam_anneal_w = 1.0
            recadam_anneal_fun = 'sigmoid'
            recadam_anneal_k = 0.5
            recadam_anneal_t0 = 250
            recadam_pretrain_cof = 5000.0
            new_model = self.model
            pretrained_model = self.pretrained_model
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type not in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": 0.0,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": 0.0,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type not in p_n]
                }
            ]
            optimizer = RecAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
        else:
            model = self.model
            
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            if self.hparams.accelerator is not None:
                optimizer = deepspeed.ops.adam.FusedAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            else: 
                optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)

        if self.hparams.use_lr_scheduling:
            if self.hparams.len_data==None:
                len_data = len(self.train_dataloader())
            else:
                len_data = int(self.hparams.len_data // self.hparams.train_batch_size)
            denomniator = (self.hparams.n_gpu * self.hparams.gradient_accumulation_steps)

            steps_per_epoch = ( len_data // denomniator ) + 1
            schedule_scale_factor = 8
            total_num_steps = ( steps_per_epoch * self.hparams.num_train_epochs ) * self.hparams.num_files * schedule_scale_factor

            print(f'total number of steps : {total_num_steps}')
            scheduler = WarmupDecayLR(optimizer, total_num_steps = total_num_steps ,warmup_max_lr = self.hparams.learning_rate, warmup_num_steps = int(total_num_steps * 0.1))
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def train_dataloader(self): 
        if self.hparams.mode=='pretrain_brute':
            train_dataset = Pretrain_Chunks(dataset_name=self.dataset_lst[self.dataset_index],tokenizer=self.tokenizer, input_length=self.hparams.max_input_length, output_length=self.hparams.max_output_length, args=self.hparams)
        else:
        train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        if self.hparams.method=='mixreview':
            #mix_len = int(len(train_dataset) * self.mix_ratio * (self.mix_decay ** self.epoch))
            mix_len = int(len(train_dataset))
            pretrain_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="pretrain", args=self.hparams, length=mix_len)
            mixed_dataset = ConcatDataset([train_dataset,pretrain_dataset])
            print("mix len is ", mix_len)
            sampler = RandomSampler(mixed_dataset)
            dataloader = DataLoader(mixed_dataset, sampler = sampler, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
            print("dataset length is ", len(dataloader.dataset))
        else:
            sampler = RandomSampler(train_dataset)
            dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        validation_dataset_unchanged = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams, lama_type='unchanged')
        validation_dataset_changed = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams, lama_type='changed')
        return [DataLoader(validation_dataset_unchanged, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False), 
                    DataLoader(validation_dataset_changed, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False),
        ]
    def test_dataloader(self):
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
        """