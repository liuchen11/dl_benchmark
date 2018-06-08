# some extra parameter parsers

import argparse

class DictParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(DictParser, self).__init__(*args, **kwargs)
        self.local_dict = {}

    def __call__(self, parser, namespace, values):

        for kv in values.split(','):
            k, v = kv.split('=')
            try:
                self.local_dict[k] = float(v)
            except:
                self.local_dict[k] = v
        setattr(namespace, self.dest, self.local_dict)
