import sys,os,re
import tempfile
import numpy as np


MAXDIM=9
RSFKEYS_READ=['in','esize','data_format','type','form','_in','infile',
        'n1','o1','d1','label1','unit1',
        'n2','o2','d2','label2','unit2',
        'n3','o3','d3','label3','unit3', 
        'n4','o4','d4','label4','unit4', 
        'n5','o5','d5','label5','unit5',
        'n6','o6','d6','label6','unit6',
        'n7','o7','d7','label7','unit7',
        'n8','o8','d8','label8','unit8',
        'n9','o9','d9','label9','unit9']
RSFKEYS_WRITE=RSFKEYS_READ[:3]
KEY_INT=["esize","n1","n2","n3","n4","n5","n6","n7","n8","n9"]
KEY_FLOAT=["o1","d1",
           "o2","d2",
           "o3","d3",
           "o4","d4",
           "o5","d5",
           "o6","d6",
           "o7","d7",
           "o8","d8",
           "o9","d9"]

def _check_key(key):
    if key not in RSFKEYS_READ + ["ax1","ax2","ax3","ax4","ax5","ax6","ax7","ax8","ax9","axes"]:
        print("Wrong keyword:",key)
        sys.exit(1)

def _set_type(key,val,quote_str=False):
    if quote_str:
        if key not in KEY_INT and key not in KEY_FLOAT:
            return quote(val)
        else:
            return val
    else:
        if key in KEY_INT:
            return int(val)
        elif key in KEY_FLOAT:
            return float(val)
        else:
            return unquote(val)


def input(filename='',mode=None,data_format=None):
    rsf=RSF(mode=mode,readonly=True,data_format=data_format)
    rsf.filename=filename
    rsf.read_rsf(filename)
    rsf.open_in()
    return rsf

def output(filename='',mode=None,data_format=None,_in=None,open_in=True,abspath=True,**kwargs):
    rsf=RSF(mode=mode,readonly=False,data_format=data_format)
    rsf.filename=filename
    for key in kwargs.keys():
        _check_key(key)
    rsf.set(**kwargs)
    rsf.wrote=False
    if filename and filename.lower() != 'stdout':
        if _in:
            rsf.set_in(_in,abspath)
        else:
            rsf.set_in(filename+'@',abspath)
        if open_in:
            rsf.open_in()
    else: # header to stdout
        if _in:
            rsf.set_in(_in,abspath)
            if open_in:
                rsf.open_in()
        else: # tempfile
            if open_in:
                rsf.open_in(tmpfile=True,abspath=abspath)
    return rsf

def fromfile(filename,key):
    with input(filename) as f:
        if type(key) == str:
            key=key.split()
        result = f.gets(key)
    return result

def tofile(filename,data,**kwargs):
    with output(filename,**kwargs) as f:
        f.write(data)

class Axis:
    def __init__(self,n=0,o=0.,d=0.,label='',unit=''):
        self.n=int(n)
        self.o=float(o)
        self.d=float(d)
        self.label=unquote(label)
        self.unit=unquote(unit)

    def to_str(self,idim):
        string='n%d=%d o%d=%s d%d=%s'%(idim,self.n,idim,self.o,idim,self.d)
        if self.label:
            string += ' label%d="%s"'%(idim,self.label)
        if self.unit:
            string += ' unit%d="%s"'%(idim,self.unit)
        return string


class RSF:
    def __init__(self,mode=None,readonly=True,data_format=None):
        self._in=""
        self._open=False
        self._data_format = data_format or 'native_float'
        self._mode = mode or self._guess_mode(readonly)
        # axes
        self.axes=[0]*MAXDIM
        for idim in range(MAXDIM):
            self.axes[idim]=Axis()
        #self.type='float' # float/int/complex - char,uchar not implemented
        #self.form='native' # ascii/native - xdr not implemented

    def _guess_mode(self,readonly):
        if readonly:
            if self.form == 'ascii':
                return 'r'
            else:
                return 'rb'
        else:
            if self.form == 'ascii':
                return 'wb' # 'wb' for np.savetxt
            else:
                return 'wb'

    def open_in(self,tmpfile=False,abspath=True):
        if tmpfile:
            self.f=tempfile.NamedTemporaryFile(mode=self._mode,suffix=".rsf@",dir="./",delete=False)
            self.set_in(self.f.name,abspath)
        else:
            self.f=open(self._in,self._mode)
        self._open=True

    # for with statement
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def get(self,key):
        if key=='in':
            return self._in
        elif key in ['esize','data_format','form','type']:
            return getattr(self,key)
        elif key=='axes':
            return self.axes
        elif key.startswith('ax'):
            idim=int(key[-1])
            return self.axes[idim-1]
        elif key=='data':
            return self.read(read_all=True)
        elif key=='header':
            return self.header_dict()
        else: # axis info - ex. n1,d1,o1,unit1,label1
            name,idim=key[:-1],int(key[-1])
            return getattr(self.axes[idim-1],name)

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,value):
        dic={key:val}
        self.set(**dic)

    def header_dict(self,contain_in=False):
        dic={}
        if contain_in:
            dic['in']=self._in
        dic.update(self.axes_dict())
        dic['data_format']=self.data_format
        dic['esize']=self.esize
        return dic

    def gets(self,keys):
        if len(keys)==1:
            return self.get(keys[0])
        else:
            return [self.get(key) for key in keys]

    def axes_dict(self):
        ax_dic={}
        for idim in range(1,self.dimension+1):
            axname="ax%d"%idim
            ax_dic[axname]=self.get(axname)
        return ax_dic

    def set_in(self,_in,abspath=True):
        self._in = os.path.abspath(_in) if abspath else _in

    def set(self,**kwargs):
        for key,val in kwargs.items():
            if key=='in' or key=='infile' or key=='_in':
                self._in=val
            elif key=='esize':
                self._esize=val
            elif key in ['data_format','form','type']:
                setattr(self,key,val)
            elif key == 'axes':
                ndim = len(val)
                for idim in range(ndim):
                    self.axes[idim]=val[idim]
            elif key.startswith('ax'):
                idim=int(key[-1])
                self.axes[idim-1]=val
            else: # axis info
                name,idim=key[:-1],int(key[-1])
                setattr(self.axes[idim-1],name,val)

    def size(self,istart=0):
        """
        Dimension size from istart+1 to max.dim
        ex. for 3D data
        size(0) returns n1*n2*n3
        size(1) returns n2*n3
        size(2) returns n3
        """
        return int(np.product(self.shape[istart:]))
#    def nelem(self):
#        ncount=1
#        for idim in range(1,MAXDIM+1):
#            n=self.get("n%d"%idim)
#            if n>0:
#                ncount *= n
#        return ncount
    @property
    def dimension(self):
        dim=0
        for idim in range(1,MAXDIM+1):
            if self.get("n%d"%idim) > 0:
                dim=idim
        return dim
    @property
    def shape(self):
        dim=self.dimension
        dimlist=["n%d"%idim for idim in range(dim,0,-1)]
        #dimlist=[]
        #for idim in range(dim,0,-1): # reverse order
        #    dimlist.append("n%d"%idim)
        return tuple(self.gets(dimlist))
    @property
    def dtype(self):
        if self.type == 'float':
            return np.float32
        elif self.type == 'int':
            return np.int32
        elif self.type == 'complex':
            return np.complex64
        else:
            print("Wrong type:",self.type)
            sys.exit(1)
    @property
    def data_format(self):
        return self._data_format
    @data_format.setter
    def data_format(self,val):
        self._data_format=val
        if not self.type.lower() in ['int','float','complex']:
            print('Wrong type:',self.type)
            sys.exit(1)
        if not self.form.lower() in ['native','ascii']:
            print('Wrong format:',self.form)
            sys.exit(1)
    @property
    def type(self):
        return self._data_format.split('_')[1].lower()
    @type.setter
    def type(self,val):
        self._data_format="%s_%s"%(self._data_format.split('_')[0],val)
    @property
    def form(self):
        return self._data_format.split('_')[0].lower()
    @form.setter
    def form(self,val):
        self._data_format="%s_%s"%(val,self._data_format.split('_')[1])
    @property
    def esize(self):
        if self.form == 'ascii':
            return 0
        if self.type in ['int','float']:
            return 4
        elif self.type in ['complex']:
            return 8

    def close(self):
        if self._open:
            self.f.close()

    def read(self,read_all=False):
        if self.form=="native":
            if read_all:
                ncount=self.size()
            else:
                ncount=self.get("n1")
            data=np.fromfile(self.f,dtype=self.dtype,count=ncount)
            if read_all:
                data.shape=self.shape
                self.close()
        elif self.form=="ascii":
            data=np.loadtxt(self.f,dtype=self.dtype)
            if data.shape==():
                data.shape=(1,)
        return data

    def traces(self):
        if self.form=='native':
            trc=self.read()
            while len(trc)>0:
                yield trc
                trc=self.read()
        elif self.form=='ascii':
            yield self.read()

    def write(self,data,close=False,fmt='%.18e'):
        self.write_rsf()
        if self.form=="native":
            data.astype(self.dtype).tofile(self.f)
        elif self.form=="ascii":
            if fmt:
                np.savetxt(self.f,data,fmt)
            else:
                np.savetxt(self.f,data)
        if close:
            self.close()

    def read_rsf(self,filename):
        header=file2str(filename)
        for key in RSFKEYS_READ:
            val,flag=fromstring(key,header)
            if flag:
                dic={key:_set_type(key,val)}
                self.set(**dic)
        infile=self.get('in')
        if not infile:
            errexit("Error: Cannot find 'in' keyword from rsf header")
        if not os.path.exists(infile):
            errexit("Error: File not exists - %s"%infile)

    def write_rsf(self):
        if self.wrote:
            return
        string=""
        for key in RSFKEYS_WRITE:
            val=self.get(key)
            val=_set_type(key,val,quote_str=True)
            string += "%s=%s\n"%(key,val)
        for idim in range(self.dimension):
            string += self.axes[idim].to_str(idim+1)+"\n"
        str2file(string,self.filename)
        self.wrote=True



def quote(str):
    return '"' + str + '"'


def unquote(str):
    if str.startswith("'") or str.startswith('"'):
        return str[1:-1]
    else:
        return str


def fromstring(key, text, un_quote=True):
    try:
        val = re.findall(key + "\s*=\s*(\S+)", text)[-1]
        # val=re.findall(key+"=(\S+)",text)[-1]
        for q in ["'", '"']:
            if val.startswith(q) and not val.endswith(q):
                val = re.findall(key + "=" + q + "(.+?)" + q, text)[-1]
        if un_quote:
            val = unquote(val)
        return val, True
    except:
        return 0, False


def file2str(filename=''):
    if filename and filename.lower() != 'stdin':
        with open(filename, 'r') as fh:
            string = fh.read()
    else:
        string = sys.stdin.read()
    return string


def str2file(string, filename=''):
    if filename and filename.lower() != 'stdout':
        with open(filename, 'w') as fh:
            fh.write(string)
    else:
        sys.stdout.write(string)


def errexit(msg, errno=1):
    warn(msg)
    sys.exit(errno)


def warn(msg):
    sys.stderr.write(msg + "\n")

