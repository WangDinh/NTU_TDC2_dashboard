"""Adapt-PCNN MLP model (physics-consistent temperature dynamics).

Ported from Adaptive-PCNN/mlp/module.py (Adapt_PC_MLP). One step:

    output = RA_T + D/division_factor - cool_effect + heat_effect
      D           = MLP(inputs_D)                       # data-driven residual
      coeff_a,b   = softplus(fc2(softplus(fc1([SA_T,SA_V,ITE_P,RA_T]))))
      cool_effect = coeff_a * (RA_T - SA_T) * SA_V / 100
      heat_effect = coeff_b * ITE_P / 100

The physics `E` term bounds each step and the only autoregressive state is
`self.last_D` (the previous prediction, fed back as RA_T). No LSTM hidden state
— that is exactly why the MLP variant stays stable over long rollouts.

Self-contained: a local DEVICE is defined here so this module never imports
rack_forecast.trainer (which has heavy import-time side effects).
"""

import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _activation(code: int) -> nn.Module:
    return {0: nn.ReLU(), 1: nn.Sigmoid(), 2: nn.Tanh()}[code]


class Adapt_PC_MLP(nn.Module):
    """Adaptive physically-consistent MLP for single-zone temperature dynamics.

    Column indices (supply_T_col, etc.) index into the input feature vector,
    whose fixed order is [SA_T, SA_V, OA_T, OA_H, ITE_P, RA_T].
    """

    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.device = device
        self.supply_T_column = cfg.supply_T_col
        self.supply_m_column = cfg.supply_m_col
        self.power_column = cfg.power_col
        self.temperature_column = cfg.temperature_col
        self.inputs_D = cfg.inputs_D
        # Position of the temperature column WITHIN the inputs_D subset, so the
        # fed-back prediction overwrites the right slot regardless of ordering.
        assert cfg.temperature_col in cfg.inputs_D, \
            'temperature_col must be part of inputs_D (the D-net must see RA_T)'
        self._temp_in_D = cfg.inputs_D.index(cfg.temperature_col)
        self.division_factor = torch.tensor(
            [float(cfg.division_factor)], dtype=torch.float32, device=device)
        self.activation_code = cfg.activation
        self.mlp_hidden_size = cfg.mlp_hidden_size
        self.mlp_num_layers = cfg.mlp_num_layers
        self.last_D = None                 # autoregressive state (previous RA_T pred)

        self._build_model()

    def _build_model(self) -> None:
        act = _activation(self.activation_code)

        # `D`: feedforward stack over inputs_D -> scalar residual.
        layers = [nn.Linear(len(self.inputs_D), self.mlp_hidden_size), act]
        for _ in range(1, self.mlp_num_layers):
            layers += [nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size), act]
        layers += [nn.Linear(self.mlp_hidden_size, 1)]
        self.model = nn.Sequential(*layers)

        # Adaptive coefficient head: [SA_T, SA_V, ITE_P, RA_T] -> (coeff_a, coeff_b).
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)

        # Xavier on weights, zeros on biases.
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.Tensor, warm_start: bool = False):
        """One step. x_: (batch, n_feat) real inputs for this timestep.

        warm_start=True uses the real RA_T (seeds last_D); otherwise the fed-back
        previous prediction replaces RA_T in both the D-input and the physics term.
        """
        x = x_[:, self.inputs_D].clone()
        SA_T = x_[:, self.supply_T_column].clone().unsqueeze(1)
        SA_V = x_[:, self.supply_m_column].clone().unsqueeze(1)
        IT_P = x_[:, self.power_column].clone().unsqueeze(1)

        if warm_start:
            RA_T = x_[:, self.temperature_column].clone().unsqueeze(1)
            self.last_D = torch.zeros((x.shape[0], 1), device=self.device)
        else:
            # Feed the previous prediction back in as the current RA_T, in both
            # the D-input (at its temperature slot) and the physics term.
            x[:, self._temp_in_D] = self.last_D[:, 0]
            RA_T = self.last_D

        # Adaptive physics coefficients (always positive via softplus).
        combined = torch.cat((SA_T, SA_V, IT_P, RA_T), dim=1)
        coeff = F.softplus(self.fc1(combined))
        coeff = F.softplus(self.fc2(coeff))
        coeff_a, coeff_b = coeff.split(1, dim=1)

        cool_effect = coeff_a * (RA_T - SA_T) * SA_V / 100
        heat_effect = coeff_b * IT_P / 100

        D = self.model(x)
        output = RA_T + D / self.division_factor - cool_effect + heat_effect
        self.last_D = output.clone()
        return output
