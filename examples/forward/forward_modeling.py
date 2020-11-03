import torch
import horovod.torch as hvd
import argparse
from pathlib import Path
from torchwi import Time2d
from torchwi.io import rsf
from torchwi.io import time_forward_distributed_dataloader

hvd.init()

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',choices=['cpu','cuda'],help='calculation device')
parser.add_argument('--fvel', type=str, default='../data/vinit.rsf',help='initial velocity model')
parser.add_argument('--fwav', type=str, default='../data/wavelet.rsf',help='source wavelet)')
parser.add_argument('--fshot',type=str, default='../data/shot.rsf',help='shot sx, sy in km')
parser.add_argument('--order',type=int,default=4,help='fdm order (space)')
parser.add_argument('--outdir',type=str, default='../forward/',help='observed data directory')
args = parser.parse_args()

nt,dt,w = rsf.fromfile(args.fwav,"n1 d1 data")
ny,nx,h,vel = rsf.fromfile(args.fvel,"n1 n2 d1 data")
nshot,sxy = rsf.fromfile(args.fshot,"n2 data")


if hvd.rank()==0:
    Path(args.outdir).mkdir(exist_ok=True)
    print("npe=%s"%hvd.size())
    print("nt=%s, dt=%s"%(nt,dt))
    print("ny=%s, nx=%s, h=%s"%(ny,nx,h))
    print("nshot=%s"%nshot)
    print("order=%d"%args.order)

DEVICE=args.device

if args.device=='cuda' and torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())

vp=torch.from_numpy(vel).to(DEVICE)
w=torch.from_numpy(w)
sxy=torch.from_numpy(sxy).to(DEVICE) # source x, y coordinate

dataloader=time_forward_distributed_dataloader(sxy)

model = Time2d(nx,ny,h,w,dt,args.order,DEVICE)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

with torch.no_grad():
    for sx,sy, ishot in dataloader:
        print("rank=%d, ishot=%d, sx=%s"%(hvd.rank(),ishot,sx))
        ry = sy
        seismo = model(vp, sx,sy,ry)
        seismo.to('cpu').detach().numpy().tofile("%s/seismo.%04d"%(args.outdir,ishot))
