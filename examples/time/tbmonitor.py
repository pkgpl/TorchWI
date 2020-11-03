import sys
import glob
import argparse
from subprocess import call

args = sys.argv

if len(args) < 1:
    print("""
    tensorboard monitor: monitor ./runs/ directory
    list: list directories
    cat: cat latest log.txt
""")

def get_dirs(logdir='runs'):
    return glob.glob(logdir+'/*')

def get_log_dir(log=''):
    if log=='':
        dirs = get_dirs()
        dirs.sort()
        return dirs[-1]
    else:
        return log


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-l','--list',action='store_const',const=True, default=False, help='list tensorboard log directory')
parser.add_argument('-t','--tail',action='store_const',const=True, default=False, help='tail(-f) latest log.txt')
#parser.add_argument('-c','--cat',action='store_const',const=True, default=False, help='cat latest log.txt')
parser.add_argument('-s','--start',action='store_const',const=True, default=False, help='start tensorboard')
parser.add_argument('--logdir', type=str, default='runs',help='tensorboard log directory')
parser.add_argument('--log',type=str,default='',help='inspect specific log instead of the latest log')
args = parser.parse_args()

if args.list:
    for d in get_dirs(args.logdir):
        print(d)
elif args.tail:
    d = get_log_dir(args.log)
    call('tail -f %s/log.txt'%d,shell=True)
elif args.start:
    cmd = 'tensorboard --logdir=%s'%args.logdir
    print(cmd)
    call(cmd,shell=True)
else:
    d = get_log_dir(args.log)
    call('cat %s/log.txt'%d,shell=True)
    print("\n//from %s/log.txt\n"%d)
