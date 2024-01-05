import os
from argparse import ArgumentParser

import torch
import lightning
import lightning.pytorch.callbacks as callbacks

from retnet import GPTR, GPTRConfig, GPTRClassifier
from lra import (ListOps, ListOpsTiny, IMDB, LRACIFAR10,
                 ParityDataset, MajorityDataset, BinaryMarkovDataset)



class LLMClassifier(lightning.LightningModule):
    def __init__(self, model, warmup_steps=1000, lr=0.05, weight_decay=0.1):
        super().__init__()
        self.model = model
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, args = batch
        lengths = args['lengths']
        logits = self.model(x, lengths)
        loss = torch.nn.CrossEntropyLoss()(logits.logits, batch[1])
        acc = (torch.argmax(logits.logits, axis=-1) == batch[1]).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, args = batch
        logits = self.model.forward(x, args['lengths'])
        loss = torch.nn.CrossEntropyLoss()(logits.logits, batch[1])
        acc = (torch.argmax(logits.logits, axis=-1) == batch[1]).float().mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", acc, prog_bar=True)
        return loss

    def create_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
    def lr_warmup_config(self):
        def warmup(step):
            """
            This method will be called for ceil(warmup_batches/accum_grad_batches) times,
            warmup_steps has been adjusted accordingly
            """
            if self.warmup_steps <= 0:
                factor = 1/((step+1)**0.5)
            else:
                factor = min(step / self.warmup_steps, 1)
                factor = factor/(max(step, self.warmup_steps)**0.5)
            return factor

        opt1 = self.create_optimizer()
        return {
            'frequency': 1,
            'optimizer': opt1,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt1, warmup),
                'interval': 'step',
                'frequency': 1,
                'name': 'lr/warmup'
            },
        }

    def configure_optimizers(self):
        return (
            self.lr_warmup_config(),
        )


def run_test(training_config,
             model_config,
             train_dataloader,
             valid_dataloader,
             val_check_interval=1.0):
    decoder_mode = training_config['decoder_mode']
    has_wg = training_config['wg']
    orig_batch_size = training_config['orig_batch_size']
    batch_split = training_config['batch_split']
    total_epochs = training_config['total_epochs']
    lr = training_config['lr']
    warmup_steps = training_config['warmup_steps']

    model = GPTRClassifier(model_config,
                           has_wg=has_wg,
                           decoder_mode=decoder_mode)
    module = LLMClassifier(model, warmup_steps=warmup_steps, lr=lr)
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')
    trainer = lightning.Trainer(max_epochs=total_epochs,
                                accumulate_grad_batches=batch_split,
                                callbacks=[lr_monitor],
                                log_every_n_steps=1,
                                val_check_interval=val_check_interval,
                                gradient_clip_val=0.5)
    trainer.fit(model=module,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader)


def test_parity(batch_split=1, num_workers=4, wg=False, decoder_mode="default"):
    orig_batch_size = 16
    batch_split = 2
    batch_size = orig_batch_size//batch_split
    # dataset = ParityDataset(maxsize=20, minsize=5, ndata=10000)
    dataset = BinaryMarkovDataset(ndata=10000,
                                  probability_retain=[0.9, 0.5],
                                  maxsize=20, minsize=5,
                                  train_split=0.8,)
    dataset.setup()
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size, num_workers=num_workers)
    model_config = GPTRConfig(vocab_size=dataset.vocab_size,
                    context_window=24,
                    nclasses=2,
                    embedding_dim=64,
                    nheads=8,
                    nlayers=6,
                    nhidden=256,
                    pdrop=0.1
                    )
    training_config = {'decoder_mode': decoder_mode,
                       'wg': wg,
                       'orig_batch_size': orig_batch_size,
                       'batch_split': batch_split,
                       'total_epochs': 2,
                       'lr': 0.05,
                       'warmup_steps': 2*1000}
    run_test(training_config, model_config, train_dataloader, valid_dataloader)


def test_parity_mini(batch_split=1, num_workers=4, wg=False, decoder_mode="default"):
    orig_batch_size = 4
    batch_split = 1
    batch_size = orig_batch_size//batch_split
    dataset = ParityDataset(maxsize=5, minsize=2, ndata=30)
    dataset.setup()
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size, num_workers=num_workers)
    model_config = GPTRConfig(vocab_size=dataset.vocab_size,
                    context_window=12,
                    nclasses=2,
                    embedding_dim=32,
                    nheads=1,
                    nlayers=1,
                    nhidden=128,
                    pdrop=0.5
                    )
    training_config = {'decoder_mode': decoder_mode,
                       'wg': wg,
                       'orig_batch_size': orig_batch_size,
                       'batch_split': batch_split,
                       'total_epochs': 1000,
                       'lr': 0.05,
                       'warmup_steps': 100}
    run_test(training_config, model_config, train_dataloader, valid_dataloader)


def test_listops_mini(batch_split=1, num_workers=4, wg=False, decoder_mode="default"):
    dataset = ListOpsTiny("listops-tiny")
    dataset.setup()

    orig_batch_size = 32
    batch_split = 1
    batch_size = orig_batch_size//batch_split
    batch_size = orig_batch_size//batch_split
    train_dataloader = dataset.train_dataloader(batch_size=batch_size,
                                                num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size,
                                              num_workers=num_workers)
    total_epochs = 5000//(len(train_dataloader)//batch_split) + 1

    train_dataloader = dataset.train_dataloader(batch_size=batch_size,
                                                num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size,
                                              num_workers=num_workers)
    model_config = GPTRConfig(vocab_size=dataset.vocab_size,
                        context_window=2048,
                        nclasses=10,
                        embedding_dim=64,
                        nheads=8,
                        nlayers=6,
                        nhidden=256,
                        pdrop=0.1
                        )
    training_config = {'decoder_mode': decoder_mode,
                       'wg': wg,
                       'orig_batch_size': orig_batch_size,
                       'batch_split': batch_split,
                       'total_epochs': total_epochs,
                       'lr': 0.05,
                       'warmup_steps': 1000}
    run_test(training_config, model_config, train_dataloader, valid_dataloader,
             val_check_interval=0.125)


def test_listops(batch_split=8, num_workers=23, wg=False, decoder_mode="default"):
    dataset = ListOps("listops-1000")
    dataset.setup()
    orig_batch_size = 32
    batch_size = orig_batch_size//batch_split
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size, num_workers=num_workers)
    total_epochs = 5000//(len(train_dataloader)//batch_split) + 1
    model_config = GPTRConfig(vocab_size=dataset.vocab_size,
                        context_window=2100,
                        nclasses=10,
                        embedding_dim=512,
                        nheads=8,
                        nlayers=6,
                        nhidden=2048,
                        pdrop=0.1
                        )
    training_config = {'decoder_mode': decoder_mode,
                       'wg': wg,
                       'orig_batch_size': orig_batch_size,
                       'batch_split': batch_split,
                       'total_epochs': total_epochs,
                       'lr': 0.05,
                       'warmup_steps': 1000}
    run_test(training_config, model_config, train_dataloader, valid_dataloader,
             val_check_interval=0.125)


def test_imdb(batch_split=32, num_workers=23, wg=False, decoder_mode="default"):
    dataset = IMDB("imdb")
    dataset.setup()
    orig_batch_size = 32
    batch_size = orig_batch_size//batch_split
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.test_dataloader(batch_size=batch_size, num_workers=num_workers)
    total_epochs = 20000//(len(train_dataloader)//batch_split) + 1
    model_config = GPTRConfig(vocab_size=dataset.n_tokens,
                    context_window=4100,
                    nclasses=2,
                    embedding_dim=512,
                    nheads=8,
                    nlayers=6,
                    nhidden=2048,
                    pdrop=0.1
                    )
    training_config = {'decoder_mode': decoder_mode,
                       'wg': wg,
                       'orig_batch_size': orig_batch_size,
                       'batch_split': batch_split,
                       'total_epochs': total_epochs,
                       'lr': 0.05,
                       'warmup_steps': 1000}
    run_test(training_config, model_config, train_dataloader, valid_dataloader)


def test_cifar10(batch_split=2, num_workers=23, wg=False, decoder_mode="default"):
    dataset = LRACIFAR10()
    dataset.setup()
    orig_batch_size = 256
    batch_size = orig_batch_size//batch_split
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.test_dataloader(batch_size=batch_size, num_workers=num_workers)
    total_epochs = 200
    model_config = GPTRConfig(vocab_size=dataset.vocab_size,
                    context_window=dataset.lmax,
                    nclasses=dataset.doutput,
                    embedding_dim=128,
                    nheads=4,
                    nlayers=3,
                    nhidden=128,
                    pdrop=0.3
                    )
    training_config = {'decoder_mode': decoder_mode,
                       'wg': wg,
                       'orig_batch_size': orig_batch_size,
                       'batch_split': batch_split,
                       'total_epochs': total_epochs,
                       'lr': 0.1,
                       'warmup_steps': 1000}
    run_test(training_config, model_config, train_dataloader, valid_dataloader)

def string_to_bool(s):
    return True if s == "True" else False


if __name__ == "__main__":
    TASKS = ['listops', 'imdb', 'cifar10', 'listops-mini', 'parity', 'parity-mini']
    DECODER_MODES = ['default', 'stirling', 'transformer']
    parser = ArgumentParser()
    parser.add_argument("--wg", default="True", choices=["False", "True"], help="Whether to include WG in RetNet")
    parser.add_argument("--decoder-mode", default="default", choices=DECODER_MODES,
                        help="Which mode for decoder to use")
    parser.add_argument("--task", default="listops", choices=TASKS,
                        help="choose an LRA dataset from available options")
    args = parser.parse_args()
    task_name = args.task
    decoder_mode = args.decoder_mode
    wg = string_to_bool(args.wg)
    if task_name == "listops":
        test_listops(wg=wg, decoder_mode=decoder_mode)
    elif task_name == "listops-mini":
        test_listops_mini(wg=wg, decoder_mode=decoder_mode)
    elif task_name == 'imdb':
        test_imdb(wg=wg, decoder_mode=decoder_mode)
    elif task_name == 'cifar10':
        test_cifar10(wg=wg, decoder_mode=decoder_mode)
    elif task_name == 'parity':
        test_parity(wg=wg, decoder_mode=decoder_mode)
    elif task_name == 'parity-mini':
        test_parity_mini(wg=wg, decoder_mode=decoder_mode)
    else:
        raise ValueError