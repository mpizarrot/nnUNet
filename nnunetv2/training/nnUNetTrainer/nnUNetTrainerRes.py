from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn
from typing import Union, List, Tuple
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetTrainerRes(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
       
        net = ResidualEncoderUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[48, 48, 96, 192, 384, 768],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=[3, 3, 3, 3, 3, 3],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[2, 2, 2, 2, 2, 2],
            num_classes=2,
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05,"affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=enable_deep_supervision
        )
        return net