import torch.nn as nn
from omegaconf import OmegaConf

class BaseAttention(nn.Module):
    @staticmethod
    def init_att_module(att_config: OmegaConf, n:int, h:int, in_feat:int, out_feat:int, debug: bool = False, **kwargs):
        # Load the configuration passed as parameter
        model_config = att_config
        att_name = model_config.name

        # This method takes as input a name of an attention mechanism, and if implemented,
        # returns an instance of the corresponding object.
        if att_name == 'vanilla_attention':
            from attentions.vanilla_attention.vanilla_attention import VanillaAttention
            att_mech = VanillaAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'rela_attention':
            from attention_zoo.attentions.rela_attention.rela_attention import RelaAttention
            att_mech = RelaAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'skyformer':
            from attention_zoo.attentions.skyformer.skyformer import SkyformerAttention
            att_mech = SkyformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'cosformer':
            from attention_zoo.attentions.cosformer.cosformer import CosformerAttention
            att_mech = CosformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'galerkin':
            from attention_zoo.attentions.galerkin_transformer.galerkin_transformer import GalerkinAttention
            att_mech = GalerkinAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'fastformer':
            from attention_zoo.attentions.fastformer.fastformer import FastFormerAttention
            att_mech = FastFormerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'linear_attention':
            from attention_zoo.attentions.linear_attention.linear_attention import LinearAttention
            att_mech = LinearAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'linformer':
            from attention_zoo.attentions.linformer.linformer import LinformerAttention
            att_mech = LinformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'performer':
            from attention_zoo.attentions.performer.performer import PerformerAttention
            att_mech = PerformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'nystromformer':
            from attention_zoo.attentions.nystromformer.nystrom_attention import NystromformerAttention
            att_mech = NystromformerAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'custom_full_attention':
            from attention_zoo.attentions.custom_full_attention.custom_full_attention import CustomFullAttention
            att_mech = CustomFullAttention(model_config, n, h, in_feat, out_feat)
        elif att_name == 'custom_nystrom_attention':
            from attention_zoo.attentions.custom_nystrom_attention.custom_nystrom_attention import \
                CustomNystromAttention
            att_mech = CustomNystromAttention(model_config, n, h, in_feat, out_feat)
        else:
            raise (
                'The name of the selected attention mechanism is not implemented. \n We support self-attention, rela_attention, skyformer, scatterbrain, cosformer, galerkin, fastformer, linear_attention, linformer, performer, nyströmformer, RNNS, custom_full_attention and custom_nystrom_attention')

        return att_mech
