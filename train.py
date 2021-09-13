import argparse
import pytorch_lightning as pl
from datamodule import ClassifyDataModule
from module import MobileNetV2
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow
from pathlib import Path
import torch
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    dict_args = vars(hparams)
    
    mlflow.set_tracking_uri("http://localhost:54849")
    mlflow.pytorch.autolog()
    
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
    with mlflow.start_run() as run:
        trainer.fit(mobilenetv2, datamod)
    trainer.save_checkpoint("checkpoints/latest.ckpt")
    # mobilenetv2.model.state_dict()
    
    metrics =  trainer.logged_metrics
    vacc, vloss, last_epoch = metrics['val_step_acc'], metrics['val_step_loss'], metrics['epoch']
    
    filename = f'mobilenet_v2-{last_epoch:02d}_acc{vacc:.4f}_loss{vloss:.4f}.pth'
    saved_filename = str(Path('weights').joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(mobilenetv2.model.state_dict(), saved_filename)
    
    
    
    
