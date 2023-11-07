from typing import List

import numpy as np
import torch
from einops import rearrange


def temporal_locality_fuse(
    f_tallies:List[List[torch.Tensor]],
    temporal_locality='both',
    fusion_method='mean',
    return_domain_names=False,
):
    """
    # Input
    - `f_tallies`:
        - stand for: [
            [source_local, source_global],
            [target_local, target_global]
        ]
        - shape:    [
            [ [B, V0, C_l], [B, V1, C_l] ],
            [ [B, V2, C_l], [B, V3, C_l] ]
        ]
    - `domains`: [D x B]

    # Output Shapes
    ```text
    |        | local                  | global                 | both                       |
    |--------|------------------------|------------------------|----------------------------|
    | none   | [B x V0 + B x V2, C_l] | [B x V1 + B x V3, C_l] | [Σ_i (B x V_i),       C_l] |
    | concat | [2 x B,     V02 x C_l] | [2 x B,     V13 x C_l] | [2 x B, (V02 + V13) x C_l] |
    | mean   | [2 x B,           C_l] | [2 x B,           C_l] | [2 x B,               C_l] |
    | linear |                        |                        |                            |
    ```
    """
    assert temporal_locality in ['both', 'local', 'global']
    assert fusion_method in ['', 'concat', 'mean']

    B = f_tallies[0][0].shape[0]
    Vs = [f.shape[1] if f is not None else 0 for f_tally in f_tallies for f in f_tally]  # [4]
    if len(Vs) < 4:  # if single domain
        Vs += [0] * (4 - len(Vs))

    if len(f_tallies[0]) == 1:  # single modal
        # TODO: Other methods
        fs = torch.cat([f for f_tally in f_tallies for f in f_tally]).mean(dim=1)
        domains = ['source'] * B + ['target'] * B

    elif temporal_locality in ['local', 'global']:
        if temporal_locality == 'local':
            fs = [f_local for f_local, _ in f_tallies]  # [[B, V0, C_l], [B, V2, C_l]]
            V_s, V_t = Vs[0], Vs[2]
        elif temporal_locality == 'global':
            fs = [f_global for _, f_global in f_tallies]  # [[B, V1, C_l], [B, V3, C_l]]
            V_s, V_t = Vs[1], Vs[3]

        # NOTE: class labels differ by batch index
        if fusion_method == '':
            fs = [rearrange(f, 'b vi cl -> (b vi) cl') for f in fs]  # [[B x V_s, C_l], [B x V_t, C_l]]
            fs = torch.cat(fs)  # [B x V_s + B x V_t, C_l]
            domains = ['source'] * (B*V_s) + ['target'] * (B*V_t)
        elif fusion_method == 'concat':
            assert V_s == V_t
            fs = [rearrange(f, 'b vi cl -> b (vi cl)') for f in fs]  # [[B, V_s x C_l], [B, V_t x C_l]]
            fs = torch.cat(fs)  # [2 x B, V_st x C_l], V_st := V_s = V_t
            domains = ['source'] * B + ['target'] * B
        elif fusion_method == 'mean':
            fs = [f.mean(dim=1) for f in fs]  # 2 x [B, C_l]
            fs = torch.cat(fs)  # [2 x B, C_l]
            domains = ['source'] * B + ['target'] * B

    elif temporal_locality == 'both':
        V_s, V_t = Vs[0] + Vs[1], Vs[2] + Vs[3]
        if fusion_method == '':
            fs = [rearrange(f, 'b vi cl -> (b vi) cl') for f_tally in f_tallies for f in f_tally]  # [[B x V0, C_l], ...]
            fs = torch.cat(fs)  # [sum_i (B x V_i), C_l]
            domains = ['source'] * (B*V_s) + ['target'] * (B*V_t)
        elif fusion_method == 'concat':
            assert V_s == V_t
            fs = [torch.cat(f_tally, dim=1) for f_tally in f_tallies]  # [[B, V0 + V1, C_l], [B, V2 + V3, C_l]]
            fs = [rearrange(f, 'b vi cl -> b (vi cl)') for f in fs]  # [[B, (V0 + V1) x C_l], [B, (V2 + V3) x C_l]]
            fs = torch.cat(fs)  # [2 x B, V_st x C_l]
            domains = ['source'] * B + ['target'] * B
        elif fusion_method == 'mean':
            fs = [torch.cat(f_tally, dim=1) for f_tally in f_tallies]  # [[B, V0 + V1, C_l], [B, V2 + V3, C_l]]
            fs = [f.mean(dim=1) for f in fs]  # 2 x [B, C_l]
            fs = torch.cat(fs)  # [2 x B, C_l]
            domains = ['source'] * B + ['target'] * B

    if return_domain_names:
        domains = np.array(domains)
        return fs, domains
    else:
        return fs
