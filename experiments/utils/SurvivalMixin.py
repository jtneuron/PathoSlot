import os
import json
import torch
import numpy as np
from sksurv.metrics import concordance_index_censored

"""
Contains metrics for survival tasks
"""

class SurvivalMixin:
    @staticmethod
    def survival_metrics(survival_events, survival_times, preds, saveto = None):
        """
        Calculate various survival metrics 
        
        Args:
            - survival_events (np.ndarray): All event indicators from test set. Shape (num_samples,)
            - survival_times (np.ndarray): All event times from test set. Shape (num_samples,)
            - preds (np.ndarray): Predicted risk scores from test set. Shape (num_samples,)
            
        Returns:
            - metrics (dict): Dictionary of metrics
        """
        # Convert survival_events to boolean
        survival_events = survival_events.astype(bool)
        
        # Assert shapes
        assert preds.ndim == 1, f"Predictions must be 1D, got shape {preds.shape}"
        num_samples = preds.shape[0]
        assert survival_events.shape == (num_samples,), f"Expected shape ({num_samples},) for survival_events, got {survival_events.shape}"
        assert survival_times.shape == (num_samples,), f"Expected shape ({num_samples},) for survival_times, got {survival_times.shape}"
        
        # Compute metrics
        metrics = {
            'cindex': concordance_index_censored(survival_events, survival_times, preds, tied_tol=1e-08)[0],
        }

        # ---- Time-dependent AUC over 36 months (monthly grid, FIXED 36 points) ----
        # 36 months -> days; use 30.4375 days/month (consistent fixed conversion)
        try:
            from sksurv.util import Surv
            from sksurv.metrics import cumulative_dynamic_auc

            # sksurv 里 event 要是 bool
            y = Surv.from_arrays(event=survival_events.astype(bool), time=survival_times)

            # month grid in days: 1..36 months (固定 36 个点，不裁剪输出)
            month_days = 30.4375
            times_full = (np.arange(1, 36 + 1) * month_days).astype(np.float32)  # (36,)

            # cumulative_dynamic_auc 只能在数据时间范围内的点上计算
            t_min = float(np.min(survival_times))
            t_max = float(np.max(survival_times))
            valid_mask = (times_full >= t_min) & (times_full <= t_max)
            times_valid = times_full[valid_mask]

            # 先建一个全 NaN 的 AUC 曲线（长度固定 36）
            auc_full = np.full((36,), np.nan, dtype=np.float32)

            if times_valid.size > 0:
                # y_train, y_test：benchmark/bootstraps 常见做法是用同一集合
                auc_valid, _mean_auc_valid = cumulative_dynamic_auc(y, y, preds, times_valid)
                auc_full[valid_mask] = np.asarray(auc_valid, dtype=np.float32)

            # 固定输出 36 点：time 永远 36；auc 不可算的点用 None
            metrics["auc_time_days_36m"] = [float(x) for x in times_full.tolist()]
            metrics["auc_by_month_36m"] = [None if np.isnan(x) else float(x) for x in auc_full.tolist()]

            # IAUC：对可算点求平均（忽略 NaN）；如果全不可算则 None
            metrics["iauc_36m"] = None if np.all(np.isnan(auc_full)) else float(np.nanmean(auc_full))

        except Exception:
            # 如果 sksurv 版本不含 cumulative_dynamic_auc 或其它异常，就跳过
            metrics["auc_time_days_36m"] = []
            metrics["auc_by_month_36m"] = []
            metrics["iauc_36m"] = None

        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, 'w') as f:
                json.dump(metrics, f, indent=4)

        return metrics

    @staticmethod
    def _calculate_risk(logits):
        """
        Take the logits of the model and calculate the risk for the patient.
        Adapted from: https://github.com/mahmoodlab/SurvPath/blob/fe4a97bf8fc57925dc81ff930ef7e1d9b2bbc83a/utils/core_utils.py#L409
        
        Args: 
            - logits (torch.Tensor): Time-bin hazard logits returned by the model.
        
        Returns:
            - risk (torch.Tensor): Scalar risk score for the patient.
        
        """
        hazards = torch.sigmoid(logits).view(-1)
        survival = torch.cumprod(1 - hazards, dim=0)
        risk = -torch.sum(survival, dim=0)
        return risk
