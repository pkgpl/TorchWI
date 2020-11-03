import sys
import logging
import socket
from datetime import datetime
import numpy as np
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from torchwi.utils.plot import perc_clip, plot_vel, plot_mig
import matplotlib.pyplot as plt

formatter = logging.Formatter('[%(asctime)s] %(message)s')

class MainLogger:
    def __init__(self, log_dir=None, name='pkgpl', hparams=None, level=logging.DEBUG):
        self.start=timer()
        # pytorch tensorboard writer - master only
        self.writer = SummaryWriter(log_dir=log_dir)
        self.get_logdir = self.writer.get_logdir
        self.add_hparams = self.writer.add_hparams
        self.add_text = self.writer.add_text
        self.add_scalar = self.writer.add_scalar
        self.add_scalars = self.writer.add_scalars
        self.add_figure = self.writer.add_figure
        self.flush = self.writer.flush
        self.loss0 = None

        self.add_text('Name', name)
        if hparams:
            self.add_text('Hyper parameters', str(vars(hparams)))

        # python logger
        self.logger = logging.getLogger(name)
        self.stream_logger = logging.getLogger('tty')

        # file logger
        fileHandler = logging.FileHandler("%s/log.txt"%(self.get_logdir()))
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        self.logger.setLevel(level)

        self.write = self.logger.debug
        self.debug = self.logger.debug
        self.info  = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # stream logger
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self.stream_logger.addHandler(streamHandler)
        self.stream_logger.setLevel(level)

    def print(self, msg):
        self.logger.debug(msg)
        self.stream_logger.debug(msg)

    def get_logdir(self):
        return self.writer.get_logdir()

    def log_loss(self, loss, epoch):
        self.add_scalar('loss', loss, epoch)
        if self.loss0 is None:
            self.loss0 = loss
        self.add_scalar('normalized_loss', loss/self.loss0, epoch)

    def log_gradient(self, grad, epoch, h,perc=99.,figsize=[15,4]):
        g = grad.to('cpu').numpy()
        g.tofile("%s/grad.%04d"%(self.get_logdir(),epoch))
        fig=plot_mig(perc_clip(g,perc),h,figsize=figsize)
        self.add_figure('gradient',fig,epoch)
        plt.close(fig)

    def log_velocity(self, vel, epoch, h,vmin=None,vmax=None,figsize=[15,4]):
        v = vel.to('cpu').detach().numpy()
        v.tofile("%s/vel.%04d"%(self.get_logdir(),epoch))
        fig = plot_vel(v,h,vmin,vmax,figsize=figsize)
        self.add_figure('velocity',fig,epoch)
        plt.close(fig)

    def output(self,model,epoch,loss,log_gradient=True):
        self.log_loss(loss,epoch)

        if log_gradient:
            grad = model.gradient()
            grad_norm = grad.norm(float('inf')).item()
            self.write("epoch %d, loss %9.3e, gnorm %9.3e"%(epoch,loss,grad_norm))
        else:
            self.write("epoch %d, loss %9.3e"%(epoch,loss))

        strftime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("%s/loss.txt"%(self.get_logdir()),'a') as fl:
            fl.write("%d %9.3e [%s]\n"%(epoch,loss,strftime))

        if epoch < model.hparams.skip_output or epoch % model.hparams.skip_output ==0:
            if log_gradient:
                self.log_gradient(grad,epoch,model.h)
            self.log_velocity(model.velocity(),epoch,model.h, model.hparams.vmin,model.hparams.vmax)
            self.flush()

#    def final(self,args,loss):
#        #hparam_dict={'lr':args.lr,'grad_norm':args.grad_norm,
#        #        'optimizer':args.optimizer,'momentum':args.momentum,
#        #        'max_epochs':args.max_epochs}
#        hparam_dict = vars(args)
#        metric_dict={'final_loss':loss}
#        self.add_hparams(hparam_dict,metric_dict)

    def progress_bar(self,count,total,status=''):
        # from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
        tavg = (timer()-self.start)/(count+1)

        bar_len=60
        frac = count/total
        filled_len = int(round(bar_len*frac))
        percents = round(100*frac,1)
        bar = '='*filled_len + '-'*(bar_len-filled_len)

        sys.stdout.write('[%s] %s%% (%s/%s,%7.2fs/it) %s\r'%(bar,percents,count,total,tavg,status))
        sys.stdout.flush()



class JobLogger():
    def __init__(self,rank,logdir):
        self.logdir=logdir
        self.rank=rank

        self.logger = logging.getLogger('job logger')
        fileHandler = logging.FileHandler("%s/log_rank.%03d"%(logdir,rank))
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        self.logger.setLevel(logging.DEBUG)

        self.hostname = socket.gethostname()
        self.ip = socket.gethostbyname(self.hostname)
        self.logger.debug("Hostname: %s, IP: %s"%(self.hostname,self.ip))

    def log(self, epoch, nshot, ishot):
        self.logger.debug("epoch %d, nshot %d, rank %d, ishot %d"%(epoch,nshot,self.rank,ishot))

