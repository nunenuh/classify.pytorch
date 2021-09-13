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
    model = MobileNetV2(**dict_args)
    
    model_checkpoint = ModelCheckpoint(monitor="val_step_loss")
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=model_checkpoint)
    trainer.fit(model, datamod)
    
