import inspect
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OCL')))

from conditioning import LangConditioning_v2
from perceptual_grouping import SlotAttentionGrouping
from neural_networks import build_two_layer_mlp

from optim.NLLSurvLoss import NLLSurvLoss


class TokenReweighting(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"TokenReweighting expects [B, N, D], got {tuple(x.shape)}")
        weight = torch.softmax(self.scorer(x), dim=1)  # [B, N, 1]
        return x * weight, weight.squeeze(-1)


class MultimodalSlotModel(nn.Module):
    def __init__(self,
                 slide_encoder,
                 post_pooling_dim,
                 task_name,
                 num_classes,
                 loss,
                 device,
                 num_surv_bins: int = 4,
                 loss_weights=(1.0, 1.0),
                 lambda_cos: float = 0.01,
                 biomarker_weights=None,
                 feature_dim: int = 768,
                 lang_dim: int = 768,
                 slot_dim: int = 256,
                 slot_iterations: int = 3):
        super().__init__()

        self.post_pooling_dim = post_pooling_dim
        self.task_name = task_name
        self.num_classes = num_classes
        self.num_surv_bins = num_surv_bins
        self.device = torch.device(device)
        if self.num_surv_bins != 4:
            raise ValueError(
                "This slot pipeline expects four discrete survival time bins. "
                "Raw survival_bins may be encoded as 0-7, but the survival head "
                "must output one hazard per time bin after survival_bins % 4."
            )

        if not isinstance(loss, (list, tuple)) or len(loss) != 2:
            raise ValueError("loss must be [cls_loss, surv_loss].")
        self.cls_loss = loss[0]
        self.surv_loss = loss[1]
        if not isinstance(self.surv_loss, NLLSurvLoss):
            raise TypeError("The second loss must be NLLSurvLoss.")

        self.w_cls, self.w_surv = loss_weights
        self.lambda_cos = lambda_cos
        self.feature_dim = feature_dim
        self.lang_dim = lang_dim
        self.slot_dim = slot_dim
        self.slot_iterations = slot_iterations

        if biomarker_weights is None:
            biomarker_weights = [1.0] * num_classes
        if len(biomarker_weights) != num_classes:
            raise ValueError(f"biomarker_weights length {len(biomarker_weights)} != num_classes {num_classes}")
        self.register_buffer("biomarker_weights", torch.tensor(biomarker_weights, dtype=torch.float32))

        self.n_slots = self.num_classes + 1
        self.survival_slot_index = self.num_classes

        self.token_reweighting = TokenReweighting(feature_dim)

        self.conditioning = LangConditioning_v2(
            object_dim=slot_dim,
            lang_dim=lang_dim,
            n_slots=self.n_slots,
        )

        ff_mlp = build_two_layer_mlp(
            input_dim=slot_dim,
            output_dim=slot_dim,
            hidden_dim=slot_dim * 4,
            initial_layer_norm=True,
            residual=True,
        )

        self.slot_attention_grouping = self._make_slot_attention_grouping(
            feature_dim=feature_dim,
            object_dim=slot_dim,
            ff_mlp=ff_mlp,
            slot_iterations=slot_iterations,
        )

        self.biomarker_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim, slot_dim),
                nn.GELU(),
                nn.Linear(slot_dim, 1),
            )
            for _ in range(self.num_classes)
        ])

        self.global_context_proj = nn.Linear(feature_dim, slot_dim)
        self.surv_context_gate = nn.Linear(slot_dim, slot_dim)
        self.surv_res_ln = nn.LayerNorm(slot_dim)
        self.survival_head = nn.Linear(slot_dim, self.num_surv_bins)

        self.to(self.device)
        self._move_losses_to_device()

    @staticmethod
    def _make_slot_attention_grouping(feature_dim: int,
                                      object_dim: int,
                                      ff_mlp: nn.Module,
                                      slot_iterations: int):
        kwargs = dict(
            feature_dim=feature_dim,
            object_dim=object_dim,
            ff_mlp=ff_mlp,
            use_projection_bias=False,
            use_implicit_differentiation=False,
        )

        try:
            params = set(inspect.signature(SlotAttentionGrouping).parameters.keys())
            for name in (
                'n_iters', 'num_iters', 'num_iterations', 'n_iterations',
                'iterations', 'iters', 'slot_iterations'
            ):
                if name in params:
                    kwargs[name] = slot_iterations
                    break
        except (TypeError, ValueError):
            pass

        module = SlotAttentionGrouping(**kwargs)
        for attr in ('n_iters', 'num_iters', 'num_iterations', 'n_iterations', 'iterations', 'iters'):
            if hasattr(module, attr):
                setattr(module, attr, slot_iterations)
                break
        return module

    def _move_losses_to_device(self):
        if isinstance(self.cls_loss, dict):
            self.cls_loss = {fold: loss_module.to(self.device) for fold, loss_module in self.cls_loss.items()}
        else:
            self.cls_loss = self.cls_loss.to(self.device)
        self.surv_loss = self.surv_loss.to(self.device)

    @staticmethod
    def _as_tokens(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"{name} must be [B, D] or [B, N, D], got {tuple(x.shape)}")

    def _get_biomarker_labels(self, labels: dict) -> torch.Tensor:
        if self.task_name in labels:
            y = labels[self.task_name]
        elif 'biomarkers' in labels:
            y = labels['biomarkers']
        else:
            raise KeyError(f"Cannot find biomarker labels by '{self.task_name}' or 'biomarkers'.")

        y = y.to(self.device).float()
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y

    def _get_fold_id(self, batch) -> int:
        if 'current_iter' not in batch:
            raise KeyError("balanced=True requires batch['current_iter'].")
        fold_id = batch['current_iter']
        if torch.is_tensor(fold_id):
            fold_id = int(fold_id.detach().cpu().item())
        return fold_id

    def forward(self, batch, current_epoch=None, output: str = 'loss'):
        if current_epoch is None:
            current_epoch = 0

        if output == 'features':
            return batch['slide']

        slide = batch['slide']

        visual_features = slide['vis_features'].to(self.device)
        text_features = slide['text_features'][:, (current_epoch % 5)].to(self.device)

        visual_tokens = self._as_tokens(visual_features, 'vis_features')
        text_tokens = self._as_tokens(text_features, 'text_features')

        if visual_tokens.size(0) != text_tokens.size(0):
            raise ValueError(
                f"Batch size mismatch: vis_features B={visual_tokens.size(0)}, "
                f"text_features B={text_tokens.size(0)}."
            )
        if visual_tokens.size(-1) != self.feature_dim or text_tokens.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got "
                f"vis D={visual_tokens.size(-1)}, text D={text_tokens.size(-1)}."
            )

        raw_tokens = torch.cat([visual_tokens, text_tokens], dim=1)
        tokens, _ = self.token_reweighting(raw_tokens)

        name_embedding = slide['name_embedding'].to(self.device)
        if name_embedding.dim() == 2:
            name_embedding = name_embedding.unsqueeze(0).expand(raw_tokens.size(0), -1, -1)

        if name_embedding.dim() != 3:
            raise ValueError(f"name_embedding must be [B, C+1, D], got {tuple(name_embedding.shape)}")
        if name_embedding.size(0) != raw_tokens.size(0):
            raise ValueError(
                f"Batch size mismatch: name_embedding B={name_embedding.size(0)}, "
                f"features B={raw_tokens.size(0)}."
            )
        if name_embedding.size(1) < self.n_slots:
            pad = torch.zeros(
                name_embedding.size(0),
                self.n_slots - name_embedding.size(1),
                name_embedding.size(-1),
                device=name_embedding.device,
                dtype=name_embedding.dtype,
            )
            name_embedding = torch.cat([name_embedding, pad], dim=1)
        elif name_embedding.size(1) > self.n_slots:
            raise ValueError(
                f"name_embedding contains too many concepts: expected {self.n_slots}, "
                f"got {name_embedding.size(1)}."
            )
        if name_embedding.size(-1) != self.lang_dim:
            raise ValueError(
                f"Expected lang_dim={self.lang_dim}, got {name_embedding.size(-1)}."
            )

        concept_mask = torch.ones(
            raw_tokens.size(0),
            self.n_slots,
            device=self.device,
            dtype=torch.float32,
        )

        conditioning, lang_proj = self.conditioning(
            name_embedding=name_embedding,
            mask=concept_mask,
            batch_size=raw_tokens.size(0),
        )

        slots, slot_attention = self.slot_attention_grouping(tokens, conditioning)

        loss_cos = 1.0 - F.cosine_similarity(
            F.normalize(slots, dim=-1),
            F.normalize(lang_proj, dim=-1),
            dim=-1,
        ).mean()

        biomarker_logits = torch.cat(
            [head(slots[:, i]) for i, head in enumerate(self.biomarker_heads)],
            dim=-1,
        )

        survival_slot = slots[:, self.survival_slot_index]
        global_context = self.global_context_proj(tokens.mean(dim=1))
        gate = torch.sigmoid(self.surv_context_gate(survival_slot))
        survival_slot = self.surv_res_ln(survival_slot + gate * global_context)
        surv_logits = self.survival_head(survival_slot)

        if output == 'logits':
            return {
                'biomarker': biomarker_logits,
                'survival': surv_logits,
            }
        if output != 'loss':
            raise ValueError(f"Unsupported output='{output}'.")

        labels = batch['labels']

        y_bio_raw = self._get_biomarker_labels(labels)
        if y_bio_raw.size(-1) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} biomarker labels, got {y_bio_raw.size(-1)}."
            )

        bio_mask = (y_bio_raw <= 1).float()
        y_bio = y_bio_raw.clone()
        y_bio[y_bio > 1] = 0.0

        if isinstance(self.cls_loss, dict):
            bce_elem = self.cls_loss[self._get_fold_id(batch)](biomarker_logits, y_bio)
        else:
            bce_elem = self.cls_loss(biomarker_logits, y_bio)

        bce_elem = bce_elem * self.biomarker_weights.to(self.device)
        bce_masked = bce_elem * bio_mask
        denom = bio_mask.sum()
        loss_bio = bce_masked.sum() / denom.clamp_min(1.0)
        if denom <= 0:
            loss_bio = bce_masked.sum() * 0.0

        y_event = labels['survival_event'].to(self.device).view(-1).long()
        y_bins_raw = labels['survival_bins'].to(self.device).view(-1).long()
        y_bins = y_bins_raw % self.num_surv_bins
        valid_surv = ((y_event == 0) | (y_event == 1)) & (y_bins_raw >= 0) & (y_bins < self.num_surv_bins)

        if valid_surv.any():
            loss_surv = self.surv_loss(
                surv_logits[valid_surv],
                y_bins[valid_surv].view(-1, 1),
                y_event[valid_surv].view(-1, 1),
            )
        else:
            loss_surv = surv_logits.sum() * 0.0

        total_loss = self.w_cls * loss_bio + self.w_surv * loss_surv + self.lambda_cos * loss_cos

        info = [{
            'loss_biomarker': float(loss_bio.detach().cpu()),
            'loss_survival': float(loss_surv.detach().cpu()),
            'loss_cos': float(loss_cos.detach().cpu()),
        }]
        return total_loss, info
