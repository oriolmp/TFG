# This script runs a simple intergration test to ensure that all the submodules work as expected
from base_attention import BaseAttention
import hydra
import torch
import omegaconf

# CONSTANTS
# This constant includes all the supported attention mechanisms
attention_names = ['vanilla_attention',
                    'skyformer',
                    'rela_attention',
                    'performer',
                    'nystromformer',
                    'linformer',
                    'linear_attention',
                    'galerkin',
                    'fastformer',
                    'custom_nystrom_attention',
                    'custom_full_attention',
                    'cosformer']

# Iterate over all the attention mechanisms and initialize them
for att_method in attention_names:
    # Create a basic configuration
    b = 5
    n = 1024
    h = 2
    in_feat = 128
    out_feat = 128

    cfg = omegaconf.OmegaConf.create({
        'ATTENTION': att_method
    })

    # Initialize the model
    print('Initializing...', att_method)
    model = BaseAttention.init_att_module(cfg, n=n, h=h, in_feat=in_feat, out_feat=out_feat)

    # Create some random input
    test_input = torch.rand(b, n, in_feat)

    # Try a forward pass
    print("Forward passing...")
    model(test_input)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
