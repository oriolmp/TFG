import torch.nn as nn
from omegaconf import OmegaConf
import sys

class BaseAttention(nn.Module):
    @staticmethod
    def init_att_module(att_config: OmegaConf, n:int, h:int, in_feat:int, out_feat:int, debug: bool = False, **kwargs):
        # Load the configuration passed as parameter
        model_config = att_config.model
        att_name = model_config.ATTENTION

        # Switch between paths in case you want to run it at the server/local
        sys.path.append(r'C:\Users\34609\VisualStudio\TFG\attention_zoo')
        # sys.path.append('/home-net/omartinez/TFG/attention_zoo')

        # This method takes as input a name of an attention mechanism, and if implemented,
        # returns an instance of the corresponding object.
        if att_name == 'vanilla_attention':
            from attentions.vanilla_attention.vanilla_attention import VanillaAttention
            att_mech = VanillaAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'cosformer':
            from attentions.cosformer.cosformer import CosformerAttention
            att_mech = CosformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'linformer':
            from attentions.linformer.linformer import LinformerAttention
            att_mech = LinformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'nystromformer':
            from attentions.nystromformer.nystrom_attention import NystromformerAttention
            att_mech = NystromformerAttention(model_config, n, h, in_feat, out_feat)
        else:
            raise (
                'The name of the selected attention mechanism is not implemented. \n We support self-attention, cosformer, linformer,  nyströmformer')

        return att_mech
