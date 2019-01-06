import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from benchmark import benchmarking
import os

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

model = model.Model(args, checkpoint)
loader = data.Data(args)
loss = loss.Loss(args, checkpoint) if not args.test_only else None

@benchmarking(team=3, task=1, model=model, preprocess_fn=None)
def inference(model, data_loader, **kwargs):
    dev = kwargs['device']
    if dev == 'cuda':
        args.cpu = False
        args.ngraph = ''
        # from trainer import Trainer
        model = model.to(torch.device('cpu' if args.cpu else 'cuda'))
        t = Trainer(args, data_loader, model, loss, checkpoint)
        metric = t.test()
    if dev == 'cpu':
        args.cpu = False
        args.ngraph = ''
        # args.ngraph = './edsr.model'
        # from trainer import Trainer
        model = model.to(torch.device('cpu' if args.cpu else 'cuda'))
        t = Trainer(args, data_loader, model, loss, checkpoint)
        metric = t.test()
    return metric

def main():
    os.nice(20)
    inference(model, loader)
    # raise ValueError("crashed because I'm a bad exception")

if __name__ == "__main__":
   try:
      main()
   except Exception as e:
      logger.exception("main crashed. Error: %s", e)
