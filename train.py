import argparse
import pytorch_lightning as pl
from datamodule import ClassifyDataModule
from module import MobileNetV2
from pytorch_lightning.callbacks import ModelCheckpoint



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    dict_args = vars(hparams)
    
    datamod = ClassifyDataModule(**dict_args)
    mobilenetv2 = MobileNetV2(**dict_args)
    
    # model_checkpoint = ModelCheckpoint(monitor="val_step_loss")
    
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoint/',
        save_top_k=1,
        filename="mobilenet_v2-{epoch:02d}-{val_step_loss:.4f}-{val_step_acc:.4f}",
        verbose=True,
        monitor='val_step_loss',
        mode='min',
    )
    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=model_checkpoint)
    trainer.fit(mobilenetv2, datamod)
    trainer.save_checkpoint("checkpoints/latest.ckpt")
    # mobilenetv2.model.state_dict()
    
    
    
    
