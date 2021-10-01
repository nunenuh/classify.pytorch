import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# from torchvision.models import 
import torchvision.models as models
import torchmetrics 


class MobileNetV2(pl.LightningModule):
    def __init__(self, pretrained=True, num_classes=2, lr=0.005, **kwargs):
        super().__init__()
        
        self.pretrained: bool = pretrained
        self.num_classes: int = num_classes
        self.learning_rate: float = lr
        
        self.model: nn.Module = self._load_mobile_net(pretrained=pretrained, num_classes=num_classes)
        
        self.trn_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.trn_loss: torchmetrics.AverageMeter  = torchmetrics.AverageMeter()
        self.val_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.val_loss: torchmetrics.AverageMeter  = torchmetrics.AverageMeter()
        
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)
        
    
    def _load_mobile_net(self, pretrained: bool, num_classes: int) -> nn.Module:
        mnv2 = models.mobilenet_v2(pretrained=pretrained)
        mnv2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(mnv2.last_channel, num_classes),
        )
    
        nn.init.normal_(mnv2.classifier[1].weight, 0, 0.01)
        nn.init.zeros_(mnv2.classifier[1].bias)
        
        return mnv2
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.model(images) # align with Attention.forward
        loss = self.criterion(preds, labels)
        return loss, preds, labels
        
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        trn_acc = self.trn_acc(preds, labels)
        trn_loss = self.trn_loss(loss)
        
        self.log('trn_step_loss', trn_loss, prog_bar=True, logger=True)
        self.log('trn_step_acc', trn_acc,  prog_bar=True, logger=True)
        
        return loss
        
    def training_epoch_end(self, outs):
        self.log('trn_epoch_acc', self.trn_acc.compute(), logger=True)
        self.log('trn_epoch_loss', self.trn_loss.compute(), logger=True)
        
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        val_acc = self.val_acc(preds, labels)
        val_loss = self.val_loss(loss)
        
        self.log('val_step_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_step_acc', val_acc,  prog_bar=True, logger=True)
        
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_epoch_acc', self.val_acc.compute(), logger=True)
        self.log('val_epoch_loss', self.val_loss.compute(), logger=True)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    
    
if __name__ == '__main__':
    mvn2 = MobileNetV2()
    batch = torch.rand(2,3,224,224)
    result = mvn2(batch)
    print(result)