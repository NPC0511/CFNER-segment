import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import scipy
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score # 序列标注评估工具
from transformers import AutoTokenizer

from src.dataloader import *
from src.llm_client import LocalLLMVerifier
from src.prototype_memory import PrototypeMemory
from src.utils import *

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

class BaseTrainer(object):
    def __init__(self, params, model, label_list):
        # parameters
        self.params = params # 配置
        self.model = model 
        self.label_list = label_list
        self.semantic_plan = None
        self.risk_graph = {}
        self.training_policy = {}
        self.prototype_memory = PrototypeMemory(
            cache_dir=getattr(params, "prototype_memory_cache_dir", "semantic_cache/prototype_memory")
        )
        self.last_prototype_anchor_loss = 0.0
        self.last_risk_filtered_pseudo_labels = 0
        self.last_risk_retained_pseudo_labels = 0
        self.last_risk_contrastive_loss = 0.0
        self.llm_verifier = None
        self.llm_verifier_stats = {
            "eligible": 0,
            "scheduled": 0,
            "budget_exhausted": 0,
            "checked": 0,
            "accepted": 0,
            "rejected": 0,
            "uncertain": 0,
            "errors": 0,
            "cache_hits": 0
        }
        self.llm_verifier_task_checks = 0
        if getattr(params, "is_use_llm_verifier", False):
            self.llm_verifier = LocalLLMVerifier(
                model_path=params.llm_verifier_model_path,
                cache_dir=params.llm_verifier_cache_dir,
                max_new_tokens=params.llm_verifier_max_new_tokens,
                temperature=params.llm_verifier_temperature,
                torch_dtype=params.llm_verifier_torch_dtype
            )
        
        # training
        self.lr = float(params.lr)
        self.mu = 0.9
        self.weight_decay = 5e-4

    def begin_task(self, task_id):
        """Reset per-task verifier accounting when a new incremental task starts."""
        self.llm_verifier_task_checks = 0
        self.llm_verifier_stats = {
            "task_id": int(task_id),
            "eligible": 0,
            "scheduled": 0,
            "budget_exhausted": 0,
            "checked": 0,
            "accepted": 0,
            "rejected": 0,
            "uncertain": 0,
            "errors": 0,
            "cache_hits": 0
        }

    def get_semantic_old_label_weights(self, refer_dims, device):
        weights = torch.ones(refer_dims, device=device)
        training_policy = getattr(self, "training_policy", None)
        if training_policy:
            old_label_weights = training_policy.get("old_label_weights", {})
        else:
            semantic_plan = getattr(self, "semantic_plan", None)
            if not semantic_plan or not semantic_plan.get("enabled", False):
                return weights
            old_label_weights = semantic_plan.get("old_label_weights", {})

        if not old_label_weights:
            return weights
        for label_name, label_weight in old_label_weights.items():
            if label_name not in self.label_list:
                continue
            label_index = self.label_list.index(label_name)
            if label_index < refer_dims:
                weights[label_index] = float(label_weight)
        return weights

    def get_prototype_anchor_weights(self, refer_dims, device):
        weights = torch.ones(refer_dims, device=device)
        training_policy = getattr(self, "training_policy", None)
        if not training_policy:
            return weights

        anchor_weights = training_policy.get("prototype_anchor_weights", {})
        if not anchor_weights:
            return weights

        for label_name, label_weight in anchor_weights.items():
            if label_name not in self.label_list:
                continue
            label_index = self.label_list.index(label_name)
            if label_index < refer_dims:
                weights[label_index] = float(label_weight)
        return weights

    def _label_to_entity_type(self, label_name):
        if "-" in label_name:
            return label_name.split("-", 1)[1]
        return label_name

    def _apply_risk_aware_pseudo_label_thresholds(self, pseudo_labels, confidence, mask_background):
        self.last_risk_filtered_pseudo_labels = 0
        self.last_risk_retained_pseudo_labels = 0
        if not getattr(self.params, "is_use_risk_aware_pseudo_label", False):
            return pseudo_labels, torch.zeros_like(mask_background, dtype=torch.bool)
        training_policy = getattr(self, "training_policy", {}) or {}
        thresholds = training_policy.get("pseudo_label_thresholds", {})
        if not thresholds:
            return pseudo_labels, torch.zeros_like(mask_background, dtype=torch.bool)

        filtered_labels = pseudo_labels.clone()
        review_mask = torch.zeros_like(mask_background, dtype=torch.bool)
        fallback_confidence = min(max(
            float(getattr(self.params, "risk_pseudo_label_fallback_confidence", 1.0)), 0.0
        ), 1.0)
        for label_name, threshold in thresholds.items():
            if label_name not in self.label_list:
                continue
            label_index = self.label_list.index(label_name)
            low_confidence = torch.logical_and(
                torch.logical_and(mask_background, filtered_labels == label_index),
                confidence < float(threshold)
            )
            # Do not remove all uncertain high-risk old labels. The teacher is
            # the only source of old supervision in the current task, so only
            # an extremely low-confidence subset becomes O.
            hard_drop = torch.logical_and(low_confidence, confidence < fallback_confidence)
            retained_for_review = torch.logical_and(low_confidence, torch.logical_not(hard_drop))
            self.last_risk_filtered_pseudo_labels += int(hard_drop.sum().item())
            self.last_risk_retained_pseudo_labels += int(retained_for_review.sum().item())
            review_mask = torch.logical_or(review_mask, low_confidence)
            filtered_labels[hard_drop] = 0
        return filtered_labels, review_mask

    def _decode_sentence(self, input_ids):
        valid_ids = []
        for token_id in input_ids.detach().cpu().tolist():
            if token_id == auto_tokenizer.pad_token_id:
                continue
            valid_ids.append(int(token_id))
        return auto_tokenizer.decode(valid_ids, skip_special_tokens=True).strip()

    def _decode_token_span(self, token_id):
        text = auto_tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
        return text

    def _apply_llm_verifier_to_pseudo_labels(self, proposed_labels, filtered_labels,
                                             rectified, review_mask):
        """Spend a bounded LLM budget only on filtered, high-risk candidates.

        A candidate is initially filtered to O by the risk-aware confidence rule.
        The verifier can restore it only after an explicit ``accept`` decision.
        Rejection, uncertainty, errors, and unscheduled candidates remain O.
        """
        if self.llm_verifier is None:
            return filtered_labels

        semantic_plan = getattr(self, "semantic_plan", None)
        if not semantic_plan or not semantic_plan.get("enabled", False):
            return filtered_labels

        training_policy = getattr(self, "training_policy", {}) or {}
        label_risks = training_policy.get("verifier_label_risks", {})
        if not label_risks:
            return filtered_labels

        batch_cap = int(getattr(self.params, "llm_verifier_max_checks_per_batch", 0))
        task_budget = int(getattr(self.params, "llm_verifier_task_budget", 0))
        if batch_cap <= 0 or task_budget <= 0:
            return filtered_labels

        min_risk = float(getattr(self.params, "llm_verifier_min_risk", 0.7))
        candidates = []
        for position in torch.nonzero(review_mask, as_tuple=False):
            batch_idx = int(position[0].item())
            token_idx = int(position[1].item())
            label_idx = int(proposed_labels[batch_idx, token_idx].item())
            if label_idx <= 0 or label_idx >= len(self.label_list) or label_idx >= rectified.shape[-1]:
                continue
            label_name = self.label_list[label_idx]
            risk = float(label_risks.get(label_name, 0.0))
            if risk < min_risk:
                continue
            confidence = float(rectified[batch_idx, token_idx, label_idx].detach().cpu().item())
            candidates.append((risk, confidence, batch_idx, token_idx, label_idx, label_name))

        self.llm_verifier_stats["eligible"] += len(candidates)
        if not candidates:
            return filtered_labels

        # High semantic risk first; among ties review the least confident token first.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        remaining_budget = max(task_budget - self.llm_verifier_task_checks, 0)
        allowed_checks = min(batch_cap, remaining_budget, len(candidates))
        self.llm_verifier_stats["budget_exhausted"] += max(len(candidates) - allowed_checks, 0)
        if allowed_checks <= 0:
            return filtered_labels

        verified_labels = filtered_labels.clone()

        for risk, confidence, batch_idx, token_idx, label_idx, label_name in candidates[:allowed_checks]:

            candidate_span = self._decode_token_span(self.inputs[batch_idx, token_idx])
            if candidate_span == "":
                continue

            sentence = self._decode_sentence(self.inputs[batch_idx])
            candidate_type = self._label_to_entity_type(label_name)
            result = self.llm_verifier.verify(
                sentence=sentence,
                candidate_span=candidate_span,
                candidate_type=candidate_type,
                retrieved_evidence=semantic_plan.get("retrieved_evidence", {}),
                teacher_label=label_name,
                teacher_confidence=confidence
            )

            self.llm_verifier_task_checks += 1
            self.llm_verifier_stats["scheduled"] += 1
            self.llm_verifier_stats["checked"] += 1
            if result.get("cache_hit", False):
                self.llm_verifier_stats["cache_hits"] += 1
            if not result.get("ok", False):
                self.llm_verifier_stats["errors"] += 1
                self.llm_verifier_stats["uncertain"] += 1
                continue

            decision = result.get("decision", "uncertain")
            if decision == "reject":
                verified_labels[batch_idx, token_idx] = 0
                self.llm_verifier_stats["rejected"] += 1
            elif decision == "accept":
                verified_labels[batch_idx, token_idx] = proposed_labels[batch_idx, token_idx]
                self.llm_verifier_stats["accepted"] += 1
            else:
                self.llm_verifier_stats["uncertain"] += 1

        return verified_labels

    
    def batch_forward(self, inputs):    
        # Compute features
        self.inputs = inputs # # (bsz, seq_len)
        self.features = self.model.forward_encoder(inputs) #  (bsz, seq_len, hidden_dim)
        # Compute logits 常规logits
        self.logits = self.model.forward_classifier(self.features)   # (bsz, seq_len, output_dim) 


    def batch_loss(self, labels):
        '''
            Cross-Entropy Loss
        '''
        self.loss = 0
        assert self.logits!=None, "logits is none!"

        # classification loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long()) # bs*seq_len, out_dim 默认自动忽略-100 label （pad、cls、sep、第二子词对应的索引）
        self.loss = ce_loss
        return ce_loss.item() 


    def _update_running_stats(self, labels_down, features, prototypes, count_features):
        cl_present = torch.unique(input=labels_down)
   
        cl_present=torch.where((cl_present < self.old_classes) & (cl_present != pad_token_label_id), cl_present, pad_token_label_id)
        cl_present = torch.unique(input=cl_present)
      
        if cl_present[0] == pad_token_label_id:
            cl_present = cl_present[1:]

        features_local_mean = torch.zeros([self.old_classes, self.params.hidden_dim]).cuda()

        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, -1, features.shape[-1])].view(features.shape[-1], -1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features

    def update_prototypes(self, train_loader):
  
        prototypes = torch.zeros([self.old_classes, self.params.hidden_dim])
        prototypes.requires_grad = False
        prototypes = prototypes.cuda()
        count_features = torch.zeros([self.old_classes], dtype=torch.long)
        count_features.requires_grad = False
        count_features = count_features.cuda()

        for X, labels in train_loader:
            X = X.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                self.refer_model.eval()
                refer_features = self.refer_model.forward_encoder(X) # (bsz,seq_len,hidden_dim)
                refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)

            probas = torch.softmax(refer_logits, dim=-1)
            _, pseudo_probas = probas.max(dim=-1)
      
            mask_bg = labels == 0
            labels[mask_bg] = pseudo_probas[mask_bg]
     
            prototypes, count_features = self._update_running_stats(labels.unsqueeze(-1).long(), refer_features, prototypes,
                                                                    count_features)

        return prototypes, count_features

    def get_prototype_weight(self, feat):
        feat_proto_distance = self.feat_prototype_distance(feat)
        weight = F.softmax(-feat_proto_distance * self.params.proto_temperature, dim=-1)
        return weight

    def feat_prototype_distance(self, feat):
        bs, seq_len, _ = feat.shape
        feat_proto_distance = -torch.ones((bs, seq_len, self.old_classes)).to(feat.device)
        for i in range(self.old_classes):
            feat_proto_distance[:, :, i] = torch.norm(self.prototypes[i].reshape(1,1,-1).expand(bs,seq_len,-1) - feat, 2, dim=-1,)
        return feat_proto_distance

    
    def before_prototype(self, train_loader):
        self.prototypes, self.count_features = self.update_prototypes(
                train_loader)

    def build_prototypes_from_labels(self, train_loader, num_classes):
        """Build first-task prototypes from gold labels before old labels disappear."""
        prototypes = torch.zeros([num_classes, self.params.hidden_dim], device="cuda")
        count_features = torch.zeros([num_classes], dtype=torch.long, device="cuda")
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                features = self.model.forward_encoder(inputs.cuda())
                labels = labels.cuda()
                for label_index in range(num_classes):
                    class_features = features[labels == label_index]
                    if class_features.numel() == 0:
                        continue
                    count_features[label_index] += class_features.shape[0]
                    prototypes[label_index] += class_features.sum(dim=0)
        nonzero = count_features > 0
        prototypes[nonzero] = prototypes[nonzero] / count_features[nonzero].unsqueeze(1)
        if was_training:
            self.model.train()
        return prototypes, count_features

    def restore_prototypes(self, payload):
        """Overlay cached prototype vectors for labels seen in an earlier task."""
        if not payload or not hasattr(self, "prototypes"):
            return 0
        saved_labels = payload.get("label_list", [])
        saved_indices = payload.get("old_label_indices", [])
        saved_vectors = payload.get("prototype_vectors")
        saved_counts = payload.get("feature_counts")
        if saved_vectors is None or saved_vectors.ndim != 2:
            return 0

        restored_count = 0
        for saved_index in saved_indices:
            if saved_index >= len(saved_labels) or saved_index >= saved_vectors.shape[0]:
                continue
            label_name = saved_labels[saved_index]
            if label_name not in self.label_list:
                continue
            current_index = self.label_list.index(label_name)
            if current_index >= self.prototypes.shape[0]:
                continue
            vector = saved_vectors[saved_index].to(self.prototypes.device)
            if vector.shape[0] != self.prototypes.shape[1]:
                continue
            self.prototypes[current_index] = vector
            if saved_counts is not None and saved_index < saved_counts.shape[0]:
                self.count_features[current_index] = saved_counts[saved_index].to(self.count_features.device)
            restored_count += 1
        return restored_count

    def prototype_anchor_loss(self, refer_dims):
        if not getattr(self.params, "is_use_prototype_anchor", False):
            return torch.tensor(0., requires_grad=True).cuda()
        if not hasattr(self, "prototypes") or self.prototypes is None:
            return torch.tensor(0., requires_grad=True).cuda()
        if self.prototypes.numel() == 0:
            return torch.tensor(0., requires_grad=True).cuda()

        proto_features = self.prototypes[:refer_dims].unsqueeze(0)
        proto_logits = self.model.forward_classifier(proto_features).squeeze(0)
        proto_targets = torch.arange(refer_dims, device=proto_logits.device).long()
        per_proto_loss = nn.CrossEntropyLoss(reduction='none')(proto_logits, proto_targets)
        anchor_weights = self.get_prototype_anchor_weights(
            refer_dims=refer_dims,
            device=proto_logits.device
        )
        return (per_proto_loss * anchor_weights).mean()

    def risk_contrastive_loss(self, original_labels):
        """Separate current new-type features from high-risk old prototypes."""
        if not getattr(self.params, "is_use_risk_contrastive", False):
            return self.features.new_tensor(0.0)
        if not hasattr(self, "prototypes") or self.prototypes is None:
            return self.features.new_tensor(0.0)

        training_policy = getattr(self, "training_policy", {}) or {}
        contrastive_pairs = training_policy.get("contrastive_pairs", [])
        if not contrastive_pairs:
            return self.features.new_tensor(0.0)

        margin = float(getattr(self.params, "risk_contrastive_margin", 0.2))
        pair_losses = []
        for pair in contrastive_pairs:
            source_type = pair.get("source")
            target_type = pair.get("target")
            risk = min(max(float(pair.get("risk", 0.0)), 0.0), 1.0)
            source_indices = [
                index for index, label_name in enumerate(self.label_list)
                if self._label_to_entity_type(label_name) == source_type
            ]
            target_indices = [
                index for index, label_name in enumerate(self.label_list)
                if self._label_to_entity_type(label_name) == target_type
                and index < self.prototypes.shape[0]
                and self.count_features[index] > 0
            ]
            if not source_indices or not target_indices:
                continue

            source_mask = torch.zeros_like(original_labels, dtype=torch.bool)
            for source_index in source_indices:
                source_mask = torch.logical_or(source_mask, original_labels == source_index)
            new_features = self.features[source_mask]
            if new_features.shape[0] < 2:
                continue

            old_prototype = self.prototypes[target_indices].mean(dim=0).detach()
            new_prototype = new_features.detach().mean(dim=0)
            similarity_to_old = F.cosine_similarity(new_features, old_prototype.unsqueeze(0), dim=-1)
            similarity_to_new = F.cosine_similarity(new_features, new_prototype.unsqueeze(0), dim=-1)
            pair_loss = F.relu(margin + similarity_to_old - similarity_to_new).mean()
            pair_losses.append(risk * pair_loss)

        if not pair_losses:
            return self.features.new_tensor(0.0)
        return torch.stack(pair_losses).mean()


    def reg_pesudo_label(self, output):

        output = torch.softmax(output, dim=-1) # (bsz, seq_len, all_dims)
        loss = -(output * torch.log(output)).mean(dim=-1)

        return loss

    def batch_loss_rdp(self, labels):
        '''
            Cross-Entropy Loss (Pseudo label) + Soft_sharp Loss (Soft label distill)
        '''

        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim # old model 输出维度

            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  


        with torch.no_grad():
            self.refer_model.eval()
            refer_features = self.refer_model.forward_encoder(self.inputs)
            refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
     
        
        mask_background = (labels < self.old_classes) & (labels != pad_token_label_id) # 0 的位置

  
        # 原型伪标签
        probs = torch.softmax(refer_logits, dim=-1) # (bs, seq_len, refer_dims)   refer_dims==old_classes

        weights = self.get_prototype_weight(refer_features)   # (bs, seq_len, old_classes)

        rectified = weights * probs
        rectified = rectified / rectified.sum(-1, keepdim=True)
        pseudo_confidence, pseudo_labels_rec = rectified.max(dim=-1)
        proposed_pseudo_labels = pseudo_labels_rec
        pseudo_labels_rec, review_mask = self._apply_risk_aware_pseudo_label_thresholds(
            pseudo_labels=pseudo_labels_rec,
            confidence=pseudo_confidence,
            mask_background=mask_background
        )

        pseudo_labels_rec = self._apply_llm_verifier_to_pseudo_labels(
            proposed_labels=proposed_pseudo_labels,
            filtered_labels=pseudo_labels_rec,
            rectified=rectified,
            review_mask=review_mask
        )

        labels[mask_background] = pseudo_labels_rec[mask_background]
        


        loss = nn.CrossEntropyLoss(reduction='none')(self.logits.permute(0,2,1), labels) # 0 新类 旧类伪标签 -100(计算的loss为0)    (bsz,seq_len)
   

        ignore_mask = (labels!=pad_token_label_id)  
        if torch.sum(ignore_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            ce_loss = loss[ignore_mask].mean()  # scalar


        old_outputs = torch.sigmoid(refer_logits) # (bsz, seq_len, refer_dims)
        old_classes = self.old_classes
        loss_soft_label = BCEWithLogitsLossWithIgnoreIndexSoftLabel(reduction='none', ignore_index=pad_token_label_id)(self.logits, old_outputs, old_classes, original_labels)
        Regularizer_soft = self.reg_pesudo_label(self.logits)

        loss_soft_label = loss_soft_label.mean()
        Regularizer_soft = Regularizer_soft.mean()

        # distill logits loss
        distill_mask = torch.logical_and(original_labels==0, original_labels!=pad_token_label_id) # 选出other class token

        if torch.sum(distill_mask.float())==0:
            distill_logits_loss = torch.tensor(0., requires_grad=True).cuda()
        else:   
            old_logits_score = F.log_softmax(
                                self.logits[distill_mask]/self.params.temperature,
                                dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)
    
            ref_old_logits_score = F.softmax(
                                refer_logits[distill_mask]/self.params.ref_temperature, 
                                dim=-1).view(-1, refer_dims)

            semantic_weights = self.get_semantic_old_label_weights(
                                refer_dims=refer_dims,
                                device=old_logits_score.device)
            kl_per_class = F.kl_div(old_logits_score, ref_old_logits_score, reduction='none')
            distill_logits_loss = (kl_per_class * semantic_weights.view(1, -1)).sum(dim=-1).mean()



        prototype_anchor_loss = self.prototype_anchor_loss(refer_dims=refer_dims)
        self.last_prototype_anchor_loss = float(prototype_anchor_loss.detach().cpu().item())
        contrastive_loss = self.risk_contrastive_loss(original_labels=original_labels)
        self.last_risk_contrastive_loss = float(contrastive_loss.detach().cpu().item())

        distill_loss = self.params.soft_param * loss_soft_label + \
                        self.params.regular_param * Regularizer_soft + \
                        self.params.distill_logits_weight * distill_logits_loss + \
                        self.params.prototype_anchor_weight * prototype_anchor_loss + \
                        self.params.risk_contrastive_weight * contrastive_loss

        self.loss = ce_loss + distill_loss

        return ce_loss.item(), distill_loss.item()

            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()        
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

    def evaluate(self, dataloader, each_class=False, entity_order=[], is_plot_hist=False, is_save_txt=False, is_plot_cm=False):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []

            for x, y in dataloader: 
                x, y = x.cuda(), y.cuda()
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
                x_list.append(x) 
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)

            
            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)   
            pred_list = torch.argmax(logits_list, dim=-1)

            ### Plot the (logits) prob distribution for each class
            if is_plot_hist: # False
                plot_prob_hist_each_class(deepcopy(y_list), 
                                        deepcopy(logits_list),
                                        ignore_label_lst=[
                                            self.label_list.index('O'),
                                            pad_token_label_id
                                        ])

          

            ### for confusion matrix visualization
            if is_plot_cm: # False
                plot_confusion_matrix(deepcopy(pred_list),
                                deepcopy(y_list), 
                                label_list=self.label_list,
                                pad_token_label_id=pad_token_label_id)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            for pred_index, gold_index in zip(pred_list, y_list):
                gold_index = int(gold_index)
                if gold_index != pad_token_label_id: # !=-100
                    pred_token = self.label_list[pred_index] # label索引转label
                    gold_token = self.label_list[gold_index]
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    pred_line.append(pred_token) 
                    gold_line.append(gold_token) 

            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            self.model.train()

            # compute overall f1 score
            # micro f1 (default)
            f1 = f1_score([gold_line], [pred_line])*100
            # macro f1 (average of each class f1)
            ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
            if not each_class: # 不打印每个类别的f1
                return f1, ma_f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, ma_f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ma_f1, ordered_f1_score_dict

    def get_entity_confusion_pairs(self, dataloader, source_entity_list, target_entity_list):
        """Count token-level predictions of a new type where the gold type is old."""
        source_entities = set(source_entity_list)
        target_entities = set(target_entity_list)
        confusion_counts = {}
        target_token_counts = {}
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.batch_forward(inputs)
                predictions = torch.argmax(self.logits, dim=-1)
                for predicted_index, gold_index in zip(predictions.view(-1), labels.view(-1)):
                    gold_index = int(gold_index.item())
                    if gold_index == pad_token_label_id:
                        continue
                    predicted_entity = self._label_to_entity_type(
                        self.label_list[int(predicted_index.item())]
                    )
                    gold_entity = self._label_to_entity_type(self.label_list[gold_index])
                    if gold_entity in target_entities:
                        target_token_counts[gold_entity] = target_token_counts.get(gold_entity, 0) + 1
                    if predicted_entity not in source_entities or gold_entity not in target_entities:
                        continue
                    pair = (predicted_entity, gold_entity)
                    confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        if was_training:
            self.model.train()
        return [
            {
                "source": source,
                "target": target,
                "count": count,
                "target_token_count": target_token_counts.get(target, 0),
                "rate": round(count / float(max(target_token_counts.get(target, 1), 1)), 6)
            }
            for (source, target), count in sorted(confusion_counts.items())
        ]

    def get_teacher_new_to_old_confusion_pairs(self, dataloader, new_entity_list, old_entity_list):
        """Measure which new gold types the old teacher maps onto old types."""
        if self.refer_model is None:
            return []
        new_entities = set(new_entity_list)
        old_entities = set(old_entity_list)
        confusion_counts = {}
        source_token_counts = {}
        self.refer_model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                predictions = torch.argmax(self.refer_model(inputs.cuda()), dim=-1)
                for predicted_index, gold_index in zip(predictions.view(-1), labels.view(-1)):
                    gold_index = int(gold_index.item())
                    if gold_index == pad_token_label_id:
                        continue
                    gold_entity = self._label_to_entity_type(self.label_list[gold_index])
                    if gold_entity not in new_entities:
                        continue
                    source_token_counts[gold_entity] = source_token_counts.get(gold_entity, 0) + 1
                    predicted_entity = self._label_to_entity_type(
                        self.label_list[int(predicted_index.item())]
                    )
                    if predicted_entity in old_entities:
                        pair = (gold_entity, predicted_entity)
                        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        return [
            {
                "source": source,
                "target": target,
                "count": count,
                "source_token_count": source_token_counts.get(source, 0),
                "rate": round(count / float(max(source_token_counts.get(source, 1), 1)), 6)
            }
            for (source, target), count in sorted(confusion_counts.items())
        ]

    def get_new_to_old_prototype_similarity_pairs(self, dataloader, new_entity_list, old_entity_list):
        """Measure contextual similarity between gold new and old entity types.

        The frozen previous encoder supplies task-specific representations while
        the development labels identify both sides of each candidate edge.  This
        is diagnostic evidence only; risk fusion decides separately whether to
        use it.
        """
        if self.refer_model is None:
            return []
        entity_types = sorted(set(new_entity_list) | set(old_entity_list))
        feature_sums = {}
        feature_counts = {}
        was_training = self.refer_model.training
        self.refer_model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                features = self.refer_model.forward_encoder(inputs.cuda())
                labels = labels.cuda()
                for entity_type in entity_types:
                    label_indices = [
                        index for index, label_name in enumerate(self.label_list)
                        if self._label_to_entity_type(label_name) == entity_type
                    ]
                    if not label_indices:
                        continue
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    for label_index in label_indices:
                        mask |= labels == label_index
                    if not torch.any(mask):
                        continue
                    selected = features[mask]
                    feature_sums[entity_type] = feature_sums.get(entity_type, 0) + selected.sum(dim=0)
                    feature_counts[entity_type] = feature_counts.get(entity_type, 0) + int(selected.shape[0])
        if was_training:
            self.refer_model.train()

        prototypes = {}
        for entity_type, feature_sum in feature_sums.items():
            count = feature_counts.get(entity_type, 0)
            if count > 0:
                prototypes[entity_type] = F.normalize(feature_sum / float(count), p=2, dim=0)

        pairs = []
        for source in sorted(new_entity_list):
            for target in sorted(old_entity_list):
                if source not in prototypes or target not in prototypes:
                    continue
                cosine = float(torch.dot(prototypes[source], prototypes[target]).item())
                pairs.append({
                    "source": source,
                    "target": target,
                    "cosine_similarity": round(cosine, 6),
                    "source_token_count": feature_counts[source],
                    "target_token_count": feature_counts[target]
                })
        return pairs

    def get_entity_context_examples(self, dataloader, entity_list, max_examples_per_type=2,
                                    context_window=12):
        """Extract compact gold-labelled contexts for instance-conditioned LLM risk scoring."""
        entity_set = set(entity_list)
        examples = {entity_type: [] for entity_type in sorted(entity_set)}
        for inputs, labels in dataloader:
            for input_row, label_row in zip(inputs.tolist(), labels.tolist()):
                for index, label_index in enumerate(label_row):
                    if label_index == pad_token_label_id:
                        continue
                    label_name = self.label_list[int(label_index)]
                    entity_type = self._label_to_entity_type(label_name)
                    if entity_type not in entity_set or len(examples[entity_type]) >= max_examples_per_type:
                        continue
                    # In BIO data, retain the first token of each entity span.
                    if label_name.startswith("I-"):
                        continue
                    left = max(0, index - context_window)
                    right = min(len(input_row), index + context_window + 1)
                    context_ids = [token_id for token_id in input_row[left:right]
                                   if token_id != auto_tokenizer.pad_token_id]
                    examples[entity_type].append({
                        "span": auto_tokenizer.decode([input_row[index]], clean_up_tokenization_spaces=True),
                        "context": auto_tokenizer.decode(context_ids, clean_up_tokenization_spaces=True)
                    })
                    if all(len(items) >= max_examples_per_type for items in examples.values()):
                        return examples
        return {entity_type: items for entity_type, items in examples.items() if items}

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "encoder": self.model.encoder.state_dict(),
            "classifier": self.model.classifier
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)

        self.model.hidden_dim = ckpt['hidden_dim']
        self.model.output_dim = ckpt['output_dim']
        missing_keys, unexpected_keys = self.model.encoder.load_state_dict(
            ckpt['encoder'],
            strict=False
        )
        self.model.classifier = ckpt['classifier']
        if len(missing_keys) > 0:
            logger.info("Encoder missing keys while loading checkpoint: %s" % str(missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Encoder unexpected keys while loading checkpoint: %s" % str(unexpected_keys))
        logger.info("Model has been load from %s" % load_path)
