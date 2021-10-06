import torch
import torch.nn as nn
import torchvision.models as models
from typing import *
import copy
import PIL
from PIL import Image
import numpy as np
from pathlib import Path
from .datamodule import transform_fn
from . import utils
import onnx
import onnxruntime as ort
import numpy as np


class MobileNetClassify(object):
    def __init__(self, weight: str = None, idx2class: str = None, pretrained: bool = True, num_classes: int = 2,
                 topk=1, device: str = 'cpu'):
        self.weight = weight
        self._idx2class = idx2class
        self.num_classes = num_classes
        self.device = device
        self.topk = topk
        self.transform = transform_fn(train=False)
        
        self.model = self._load_mobile_net(pretrained=pretrained, num_classes=num_classes)
        self._init_model()
        self._init_idx2class()
        
    def _init_model(self):
        if self.weight:
            self.state_dict = self._load_weight(self.weight)
            self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        
    def _init_idx2class(self):
        if type(self._idx2class) == type(None):
            idx2class = {0: 'idcard', 1: 'other'}
        elif type(self._idx2class) == str:
            loaded_json = utils.load_json(self._idx2class)
            idx2class = dict()
            for key, val in loaded_json.items():
                idx2class[int(key)] = val
        elif type(self._idx2class) == dict:
            idx2class = self._idx2class
        else:
            idx2class = {0: 'idcard', 1: 'other'}
            
        self.idx2class: dict = copy.deepcopy(idx2class)
        
        
    def _load_mobile_net(self, pretrained: bool, num_classes: int) -> nn.Module:
        mnv2 = models.mobilenet_v2(pretrained=pretrained)
        mnv2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(mnv2.last_channel, num_classes),
        )
    
        nn.init.normal_(mnv2.classifier[1].weight, 0, 0.01)
        nn.init.zeros_(mnv2.classifier[1].bias)
        
        return mnv2
    
    def _load_weight(self, weight: str, key=None) -> OrderedDict:
        state_dict = torch.load(weight, map_location=torch.device('cpu'))
        return state_dict
    
    def _load_image(self, impath: str):
        impath = Path(impath)
        if impath.exists():
            image = Image.open(impath)
            image.convert("RGB")
            return image
        else:
            raise ValueError()
        
        
    def _clean_result(self, result: Tuple[torch.Tensor, torch.Tensor]):
        probability, classes = result
        probability, classes = probability.squeeze().numpy(), classes.squeeze().numpy()
        probability = probability.astype(np.float).tolist()
        classes = classes.astype(np.int).tolist()
        if self.topk>1:
            output = []
            for idx, (prob, clz) in enumerate(zip(probability, classes)):
                out = {'class': self.idx2class[clz], 'confidence': f'{prob*100:.0f}%'}
                output.append(out)
            return output
        else:
#             print(classes)
            return {'class': self.idx2class[classes], 'confidence': f'{probability*100:.0f}%'}
    
    def _predict(self, image: Union[str, Image.Image]):
        if type(image)==str:
            image = self._load_image(image)
        
        image = self.transform(image)
        image = image.unsqueeze(dim=0)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(image)
            output = torch.log_softmax(preds, dim=1)
            ps = torch.exp(output)
            result = ps.topk(self.topk, dim=1, largest=True, sorted=True)
            result = self._clean_result(result)
            
        return result
    
    def predict(self, image: Union[str, Image.Image]):
        result = self._predict(image)
        return result
        
        

class MobileNetClassifyOnnx(object):
    def __init__(self, weight: str = None, idx2class: str = None, 
                 num_classes: int = 2, topk=1, device: str = 'cpu'):
        self.weight = weight
        self._idx2class = idx2class
        self.num_classes = num_classes
        self.device = device
        self.topk = topk
        self.transform = transform_fn(train=False)
        
        self._load_check_model()
        self._init_session()
        self._init_idx2class()
    
    @property
    def _providers(self):
        return {"cpu":'CPUExecutionProvider', "cuda": 'CUDAExecutionProvider'}
        
    def _init_session(self):
        if self.weight:
            self.session = ort.InferenceSession(self.weight)
            
        if ort.get_device() == 'GPU':
            providers = [self._providers.get(self.device, "cpu")]
            self.session.set_providers(providers)
        
    def _init_idx2class(self):
        if type(self._idx2class) == type(None):
            idx2class = {0: 'idcard', 1: 'other'}
        elif type(self._idx2class) == str:
            loaded_json = utils.load_json(self._idx2class)
            idx2class = dict()
            for key, val in loaded_json.items():
                idx2class[int(key)] = val
        elif type(self._idx2class) == dict:
            idx2class = self._idx2class
        else:
            idx2class = {0: 'idcard', 1: 'other'}
            
        self.idx2class: dict = copy.deepcopy(idx2class)
        
    
    def _load_check_model(self):
        self.onnx_model = onnx.load(self.weight)
        onnx.checker.check_model(self.onnx_model)
        
    def _to_numpy(self, tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy() 
        else:
            return tensor.cpu().numpy()
    
    
    def _load_image(self, impath: str):
        impath = Path(impath)
        if impath.exists():
            image = Image.open(impath)
            image.convert("RGB")
            return image
        else:
            raise ValueError()
        
        
    def _clean_result(self, result: Tuple[torch.Tensor, torch.Tensor]):
        probability, classes = result
        probability, classes = probability.squeeze().numpy(), classes.squeeze().numpy()
        probability = probability.astype(np.float).tolist()
        classes = classes.astype(np.int).tolist()
        if self.topk>1:
            output = []
            for idx, (prob, clz) in enumerate(zip(probability, classes)):
                out = {'class': self.idx2class[clz], 'confidence': f'{prob*100:.0f}%'}
                output.append(out)
            return output
        else:
            return {'class': self.idx2class[classes], 'confidence': f'{probability*100:.0f}%'}
        
    def _onnx_predict(self, inputs):
        sess_inputs = {self.session.get_inputs()[0].name: inputs}
        ort_outs = self.session.run(None, sess_inputs)
        onnx_predict = ort_outs[0]
        return onnx_predict
    
    def _preprocess(self, image: Union[str, Image.Image]):
        if type(image)==str: image = self._load_image(image)
        image = self.transform(image)
        image = image.unsqueeze(dim=0)
        inputs = self._to_numpy(image)
        
        return inputs
    
    def _postprocess(self, prediction:np.ndarray):
        preds = torch.from_numpy(prediction)
        output = torch.log_softmax(preds, dim=1)
        ps = torch.exp(output)
        result = ps.topk(self.topk, dim=1, largest=True, sorted=True)
        clean_result = self._clean_result(result)
        return clean_result
    
    def _predict(self, image: Union[str, Image.Image]):
        inputs = self._preprocess(image)
        predict = self._onnx_predict(inputs)
        result = self._postprocess(predict)
        return result
    
    def predict(self, image: Union[str, Image.Image]):
        result = self._predict(image)
        return result
    
if __name__ == "__main__":
    mnet = MobileNetClassify(weight='weights/mobilenet_v2-09_acc0.9989_loss0.0051.pth',topk=1)
    # impath = 'images/samples/idcard/55bgkl_561809_image.jpg'
    impath = 'images/samples/other/000000069480.jpg'
    result = mnet.predict(impath)
    print(result)