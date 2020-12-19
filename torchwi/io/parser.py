import configparser
import yaml


class CFGParser:
    def __init__(self,fcfg,extended_interpolation=False):
        if extended_interpolation:
            self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        else:
            self.config = configparser.ConfigParser()
        self.config.read(fcfg)

    def parse(self,section):
        return {k: yaml.safe_load(v) for k, v in self.config.items(section)}

    def parse_sections(self,*sections):
        d={}
        for section in sections:
            d.update(self.parse(section))
        return d

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
