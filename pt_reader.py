import torch
import argparse
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint', type=str, required=True)

args = argparser.parse_args()

checkpoint = torch.load(args.checkpoint,map_location='cpu')
print(checkpoint['args'])
print(checkpoint['time_tot'])