import torch
import horovod.torch as hvd
import argparse
import numpy as np
from pathlib import Path
from torchwi import Time2d
from torchwi.io import rsf,time_distributed_dataloader
from torchwi.io import MainLogger,JobLogger
from torchwi.parameter import VelocityParameter

hvd.init()
torch.manual_seed(0)
np.random.seed(0)

class TimeInv(torch.nn.Module):

    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams
        # read velocity
        self.ny,self.nx,self.h,vinit = rsf.fromfile(self.hparams.fvel,"n1 n2 d1 data")
        #vel = torch.from_numpy(vinit).to(self.hparams.device)
        # read source wavelet
        self.nt,self.dt,w = rsf.fromfile(self.hparams.fwav,"n1 d1 data")
        self.w = torch.from_numpy(w)
        # wavefield modeling
        self.velgen=VelocityParameter(torch.from_numpy(vinit).to(self.hparams.device),
                                    self.hparams.vmin,self.hparams.vmax)
        self.modeling = Time2d(self.nx,self.ny,self.h,self.w,self.dt,self.hparams.order,self.hparams.device)
        self.gnorm0 = None

    def forward(self, sx,sy,ry):
        return self.modeling(self.velgen(), sx,sy,ry)

    def gradient(self):
        return self.velgen.gradient()

    def velocity(self):
        return self.velgen()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optim= torch.optim.Adam(self.parameters(), lr=self.hparams.lr)#*hvd.size())
        elif self.hparams.optimizer == 'nag':
            optim= torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=True)
        elif self.hparams.optimizer == 'momentum':
            optim= torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            optim= torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        # horovod
        backward_passes_per_step = 10000 #int((self.nshot-1)/hvd.size()+1)
        return hvd.DistributedOptimizer(optim,
                named_parameters=self.named_parameters(),
                backward_passes_per_step=backward_passes_per_step,
                op=hvd.Sum)

    def norm_grad(self):
        if self.hparams.grad_norm == 'none':
            pass
        else:
            grad = hvd.allreduce(self.velgen.gradient(),op=hvd.Sum)
            if self.gnorm0 is None: # first epoch
                self.gnorm0 = grad.norm(float('inf'))
            self.velgen.normalize_gradient(self.gnorm0)

    @staticmethod
    def add_model_spectific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],formatter_class=argparse.ArgumentDefaultsHelpFormatter,add_help=False)
        # data, modeling
        parser.add_argument('--fvel', type=str, default='../data/vinit.rsf',help='initial velocity model')
        parser.add_argument('--fwav', type=str, default='../data/wavelet.rsf',help='source wavelet)')
        parser.add_argument('--fshot',type=str, default='../data/shot.rsf',help='shot sx, sy in km')
        parser.add_argument('--ftrue',type=str, default='../forward/seismo.',help='observed data: shot number will be appended [04d]')
        parser.add_argument('--order',type=int,default=4,help='fdm order (space)')
        # inversion
        parser.add_argument('--vmin',type=float,default=1.5,help='minimum velocity for clipping')
        parser.add_argument('--vmax',type=float,default=5.5,help='maximum velocity for clipping')
        parser.add_argument('--grad_norm',type=str,default='first',choices=['first','none'], help='normalize gradient using the infinity norm of the first gradient')
        # optimizer
        parser.add_argument('--lr',type=float,default=0.02,help='learning rate')
        parser.add_argument('--optimizer', type=str, default='adam',choices=['adam','nag','momentum','gd'],help='optimizer: Adam|Nesterov AG|Momentum(m=0.9)|Gradient Descent|')
        parser.add_argument('--momentum',type=float,default=0.9,help='momentum value for NAG and Momentum optimizer')
        return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # program level args
    parser.add_argument('--name',type=str,default='inv2d',help='name of the experiment')
    parser.add_argument('--device',type=str,default='cpu',choices=['cpu','cuda'],help='calculation device')
    # model specific args
    parser = TimeInv.add_model_spectific_args(parser)
    # trainer args
    parser.add_argument('--skip_output',type=int,default=10,help='epoch skip for grad/vel output')
    parser.add_argument('--max_epochs',type=int,default=501,help='total number of epochs')
    args = parser.parse_args()

    if args.device=='cuda' and torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    # logging
    logdir=''
    if hvd.rank()==0:
        logger= MainLogger(name=args.name)
        logdir = logger.get_logdir()
        logger.print("Inversion start: %s (log_dir=%s)"%(args.name, logdir))
        logger.print("hyper parameters: %s"%args)
    logdir = hvd.broadcast_object(logdir, 0)
    joblogger=JobLogger(hvd.rank(),logdir)

    model = TimeInv(args)
    model.to(args.device)

    nshot,sxy = rsf.fromfile(args.fshot,"n2 data")
    dataloader= time_distributed_dataloader(args.ftrue, torch.from_numpy(sxy))
    optimizer = model.configure_optimizers()

    model.train()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(args.max_epochs+1):

        total_loss=torch.tensor(0.)
        optimizer.zero_grad()

        for sx,sy,true,ishot in dataloader:
            joblogger.log(epoch,nshot,ishot)

            true = true.to(args.device)
            ry = sy
            frd = model(sx,sy,ry)
            loss = 0.5*torch.sum((frd-true.view_as(frd))**2)

            loss.backward()
            total_loss.add_(loss.item())

        total_loss = hvd.allreduce(total_loss,op=hvd.Sum)

        # update
        model.norm_grad()
        optimizer.step()

        # output
        if hvd.rank()==0:
            if epoch==0:
                total_loss0 = total_loss.item()
            grad_norm = model.gradient().norm(float('inf')).item()
            logger.progress_bar(epoch,args.max_epochs,"norm loss=%9.3e, gnorm=%9.3e"%(total_loss.item()/total_loss0,grad_norm))
            logger.output(model, epoch, total_loss.item())

    # end
    if hvd.rank()==0:
        logger.print("\nInversion End")
