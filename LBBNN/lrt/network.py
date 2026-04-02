from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BayesianLinearLRT


class BayesianNetworkLRT(nn.Module):
    def __init__(
            self, 
            dim: int, 
            p: int, 
            hidden_layers: int, 
            a_prior: float = 0.05, 
            std_prior: float = 2.5, 
            classification: bool = True, 
            n_classes: int = 1, 
            act_func: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid, 
            lower_init_lambda: float = -10, 
            upper_init_lambda: float = -7, 
            high_init_covariate_prob: bool = False) -> None:
        super().__init__()
        self.p = p
        self.classification = classification
        self.multiclass = n_classes > 1
        self.act = act_func
        nr_var = p if high_init_covariate_prob else None
        self.linears = nn.ModuleList(                      # input layer
            [BayesianLinearLRT(
                p, 
                dim, 
                a_prior=a_prior, 
                std_prior=std_prior, 
                lower_init_lambda=lower_init_lambda, 
                upper_init_lambda=upper_init_lambda, 
                p=nr_var)])
        self.linears.extend(                               # hidden layers
            [
                BayesianLinearLRT(
                    dim + p, 
                    dim, a_prior=a_prior, 
                    std_prior=std_prior, 
                    lower_init_lambda=lower_init_lambda, 
                    upper_init_lambda=upper_init_lambda, 
                    p=nr_var) 
            for _ in range(hidden_layers - 1)])
        self.linears.append(                               # output layer
            BayesianLinearLRT(
                dim + p, 
                n_classes, 
                a_prior=a_prior, 
                std_prior=std_prior, 
                lower_init_lambda=lower_init_lambda, 
                upper_init_lambda=upper_init_lambda, 
                p=nr_var))
        
        self.loss = nn.NLLLoss(reduction='sum') if (classification and self.multiclass) else (nn.BCELoss(reduction='sum') if classification else nn.MSELoss(reduction='sum'))

    def forward(
            self, 
            x: torch.Tensor, 
            sample: bool = False, 
            ensemble: bool = True, 
            calculate_log_probs: bool = False, 
            post_train: bool = False) -> torch.Tensor:
        x_input = x.view(-1, self.p)
        x_hidden = self.act(self.linears[0](x_input, ensemble, sample, calculate_log_probs, post_train))
        i = 1
        for layer in self.linears[1:-1]:
            x_hidden = self.act(layer(torch.cat((x_hidden, x_input), dim=1), ensemble, sample, calculate_log_probs, post_train))
            i += 1
        logits = self.linears[i](torch.cat((x_hidden, x_input), dim=1), ensemble, sample, calculate_log_probs, post_train)
        if self.classification:
            return F.log_softmax(logits, dim=1) if self.multiclass else torch.sigmoid(logits)
        return logits

    def forward_preact(
            self, 
            x: torch.Tensor, 
            sample: bool = False, 
            ensemble: bool = False, 
            calculate_log_probs: bool = False, 
            post_train: bool = False) -> torch.Tensor:
        x_input = x.view(-1, self.p)
        x_hidden = self.act(self.linears[0](x_input, ensemble, sample, calculate_log_probs, post_train))
        i = 1
        for layer in self.linears[1:-1]:
            x_hidden = self.act(layer(torch.cat((x_hidden, x_input), dim=1), ensemble, sample, calculate_log_probs, post_train))
            i += 1
        return self.linears[i](torch.cat((x_hidden, x_input), dim=1), ensemble, sample, calculate_log_probs, post_train)

    def kl(self) -> torch.Tensor:
        return sum(layer.kl for layer in self.linears)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self(x, ensemble=False)
            if self.classification:
                return out.argmax(dim=1) if self.multiclass else (out >= threshold).float().view(-1)
            return out.view(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self(x, ensemble=False)
            return torch.exp(out) if self.multiclass else out


class InputSkipLRTNetwork(BayesianNetworkLRT):
    pass
