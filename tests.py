import os
from argparse import ArgumentParser

import torch
import lightning

from retnet import GPTR, GPTRConfig, GPTRClassifier
from lra import ListOps, IMDB



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
                factor = 1
            else:
                factor = min(step / self.warmup_steps, 1)
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


def test_listops(batch_split=8, num_workers=23, wg=False):
    dataset = ListOps("listops-1000")
    dataset.setup()
    orig_batch_size = 32
    batch_size = orig_batch_size//batch_split
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size, num_workers=num_workers)
    total_epochs = 5000//(len(train_dataloader)//batch_split) + 1
    config = GPTRConfig(vocab_size=dataset.vocab_size,
                    context_window=None,
                    nclasses=10,
                    embedding_dim=512,
                    nheads=8,
                    nlayers=6,
                    nhidden=2048
                    )
    model = GPTRClassifier(config, has_wg=wg)
    module = LLMClassifier(model, warmup_steps=0)
    trainer = lightning.Trainer(max_epochs=2, accumulate_grad_batches=8)
    trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

def test_imdb(batch_split=16, num_workers=23, wg=False):
    dataset = IMDB("imdb")
    dataset.setup()
    orig_batch_size = 32
    batch_size = orig_batch_size//batch_split
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = dataset.val_dataloader(batch_size=batch_size, num_workers=num_workers)
    total_epochs = 20000//(len(train_dataloader)//batch_split) + 1
    config = GPTRConfig(vocab_size=dataset.n_tokens,
                    context_window=None,
                    nclasses=2,
                    embedding_dim=512,
                    nheads=8,
                    nlayers=6,
                    nhidden=2048
                    )
    model = GPTRClassifier(config, has_wg=wg)
    module = LLMClassifier(model, warmup_steps=1000*batch_split)
    trainer = lightning.Trainer(max_epochs=2, accumulate_grad_batches=8)
    trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    TASKS = ['listops', 'imdb']
    parser = ArgumentParser()
    parser.add_argument("--wg", default="False", choices=["False", "True"], help="Whether to include WG in RetNet")
    parser.add_argument("--task", default="listops", choices=TASKS,
                        help="choose an LRA dataset from available options")
    args = parser.parse_args()
    task_name = args.task
    wg = args.wg
    wg = True if wg == "True" else False
    if task_name == "listops":
        test_listops(wg=wg)
    elif task_name == 'imdb':
        test_imdb(wg=wg)
