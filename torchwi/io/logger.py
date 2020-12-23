import sys
import logging
import socket
from datetime import datetime
import numpy as np
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from torchwi.utils.plot import perc_clip, plot_vel, plot_mig
import matplotlib.pyplot as plt
import yaml

formatter = logging.Formatter('[%(asctime)s] %(message)s')

class MainLogger:
    def __init__(self, log_dir=None,comment='', name='pkgpl', level=logging.DEBUG):
        self.start=timer()
        # pytorch tensorboard writer - master only
        self.writer = SummaryWriter(log_dir=log_dir,comment=comment)
        self.get_logdir = self.writer.get_logdir
        self.add_hparams = self.writer.add_hparams
        self.add_text = self.writer.add_text
        self.add_scalar = self.writer.add_scalar
        self.add_scalars = self.writer.add_scalars
        self.add_figure = self.writer.add_figure
        self.flush = self.writer.flush
        self.loss0 = dict()

        self.add_text('Name', name)

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

    def log_hparams(self, name, hparams_dict):
        self.print("%s:\n%s"%(name,yaml.dump(hparams_dict)))
        self.add_text(name, str(hparams_dict))

    def log_loss(self, loss, epoch, name='loss',filename=None, add_figure=True,log_norm=True):
        if filename is None:
            filename = name+'.txt'
        # file output
        strftime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("%s/%s"%(self.get_logdir(),filename),'a') as fl:
            fl.write("%d %9.3e [%s]\n"%(epoch,loss,strftime))
        # tensorboard
        if add_figure:
            self.add_scalar(name, loss, epoch)
            if log_norm:
                if self.loss0.get(name) is None:
                    self.loss0[name] = loss
                self.add_scalar('normalized_%s'%name, loss/self.loss0[name], epoch)

    def log_gradient(self, grad, epoch, h, filename='grad.', add_figure=True,figurename='gradient',perc=99.,figsize=[15,4]):
        # file output
        g = grad.to('cpu').numpy()
        g.tofile("%s/%s%04d"%(self.get_logdir(),filename,epoch))
        # tensorboard
        if add_figure:
            fig=plot_mig(perc_clip(g,perc),h,figsize=figsize)
            self.add_figure(figurename,fig,epoch)
            plt.close(fig)

    def log_velocity(self, vel, epoch, h, filename='vel.', add_figure=True,figurename='velocity',vmin=None,vmax=None,figsize=[15,4]):
        # file output
        v = vel.to('cpu').detach().numpy()
        v.tofile("%s/%s%04d"%(self.get_logdir(),filename,epoch))
        # tensorboard
        if add_figure:
            fig = plot_vel(v,h,vmin,vmax,figsize=figsize)
            self.add_figure(figurename,fig,epoch)
            plt.close(fig)

    def output(self,model,epoch,loss,log_gradient=True):
        self.log_loss(loss,epoch)

        if log_gradient:
            grad = model.gradient()
            grad_norm = grad.norm(float('inf')).item()
            self.write("epoch %d, loss %9.3e, gnorm %9.3e"%(epoch,loss,grad_norm))
        else:
            self.write("epoch %d, loss %9.3e"%(epoch,loss))

        if epoch < model.hparams.skip_output or epoch % model.hparams.skip_output ==0:
            if log_gradient:
                self.log_gradient(grad,epoch,model.h)
            self.log_velocity(model.velocity(),epoch,model.h, vmin=model.hparams.vmin, vmax=model.hparams.vmax)
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
        frac = count/(total-1)
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

