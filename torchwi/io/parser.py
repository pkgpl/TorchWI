import configparser
import yaml
from argparse import Namespace


class CFGParser:
    def __init__(self,fcfg,extended_interpolation=False,namespace=False,default={}):
        if extended_interpolation:
            self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        else:
            self.config = configparser.ConfigParser()
        self.config.read(fcfg)
        self.namespace = namespace
        self.default = default

    def parse(self,section):
        d={k: yaml.safe_load(v) for k, v in self.config.items(section)}
        return d

    def parse_sections(self,sections):
        d=self.default
        #d={}
        for section in sections:
            d.update(self.parse(section))
        if self.namespace:
            return self.dict_to_namespace(d)
        else:
            return d

    def dict_to_namespace(self,d):
        return Namespace(**d)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


def parse_args(fcfg,sections,**kwargs):
    hparams = CFGParser(fcfg,extended_interpolation=True,namespace=True,default=kwargs).parse_sections(sections)
    return hparams

def parse_list(val,typ=int):
    if type(val) == typ:
        return [val]
    else:
        return [typ(ival) for ival in val.split(',')]
