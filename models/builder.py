import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

import torch.nn as nn
from torchvision import transforms

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        # c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1)

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    # elif 'dsmil' in model_name:
    #     if 'camel' in model_name:
    #         weight_path = "/home/z/zeyugao/PreModel/dsmil/model-v2-camel.pth"
    #     elif 'lung' in model_name:
    #         weight_path = "/home/z/zeyugao/PreModel/dsmil/model-v1-lung.pth"
    #     else:
    #         raise NotImplementedError('model {} no weights'.format(model_name))

    #     norm=nn.InstanceNorm2d
    #     resnet = models.resnet18(pretrained=False, norm_layer=norm)
    #     resnet.fc = nn.Identity()
    #     model = IClassifier(resnet, 512, output_class=1).cuda()

    #     state_dict_weights = torch.load(weight_path)
    #     for i in range(4):
    #         state_dict_weights.popitem()
    #     state_dict_init = model.state_dict()
    #     new_state_dict = OrderedDict()
    #     for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
    #         name = k_0
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict, strict=False)
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        # assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
        target_img_size = 448
        
    elif model_name == 'virchow2':
        try:
            from timm.layers import SwiGLUPacked
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use Virchow2")
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    elif model_name == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif model_name == 'hibou_l':
        from transformers import AutoModel
        model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
    elif model_name == 'h_optimus_0':
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)

    if model_name in MODEL2CONSTANTS.keys():
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    elif model_name == 'virchow2':
        try:
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use Virchow2")
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == 'gigapath':
        img_transforms = transforms.Compose(
                        [
                            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ]
                    )
    elif model_name == 'hibou_l':
        img_transforms = transforms.Compose([
                            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816]),
                        ])
    elif model_name == 'h_optimus_0':
        img_transforms = transforms.Compose([
                            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=(0.707223, 0.578729, 0.703617), 
                                std=(0.211883, 0.230117, 0.177517)
                            ),
                        ])
    else:
        img_transforms = get_eval_transforms(mean=None,
                                            std=None,
                                            target_img_size = target_img_size)

    return model, img_transforms