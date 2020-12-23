import torch

class BaseParameter(torch.nn.Module):
    def __init__(self, vel, vmin=None, vmax=None):
        super().__init__()
        self.vmin=vmin
        self.vmax=vmax
        self.par=torch.nn.Parameter(self.vel_to_par(vel))
        if vmin and vmax:
            self.pmin=min(self.vel_to_par(vmin),self.vel_to_par(vmax))
            self.pmax=max(self.vel_to_par(vmin),self.vel_to_par(vmax))

    def gradient(self):
        return self.par.grad

    def gradient_times(self,multiplier):
        self.par.grad *= multiplier

    def grad_norm(self):
        return self.par.grad.norm(float('inf'))

    def normalize_gradient(self, gnorm0):
        self.par.grad /= gnorm0

    def vel_to_par(self,vel):
        return vel

    def par_to_vel(self,par):
        return par

    def vel(self):
        return self.par_to_vel(self.par)

    def pclip(self):
        if self.vmin and self.vmax:
            with torch.no_grad():
                self.par[:] = torch.clamp(self.par, self.pmin, self.pmax)

    def forward(self):
        self.pclip()
        return self.vel()

    def report(self):
        return "Parameter: %s"%(self.__class__.__name__)


class VelocityParameter(BaseParameter):
    def __init__(self, vel, vmin=None, vmax=None):
        super().__init__(vel, vmin, vmax)


class SlothParameter(BaseParameter):
    def __init__(self, vel, vmin=None, vmax=None):
        super().__init__(vel, vmin, vmax)

    def vel_to_par(self,vel):
        return 1./vel**2

    def par_to_vel(self,par):
        return torch.sqrt(1./par)


class SlownessParameter(BaseParameter):
    def __init__(self, vel, vmin=None, vmax=None):
        super().__init__(vel, vmin, vmax)

    def vel_to_par(self,vel):
        return 1./vel

    def par_to_vel(self,par):
        return 1./par

