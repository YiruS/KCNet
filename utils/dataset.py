from __future__ import print_function

import os
import os.path
import sys
import argparse
import pickle
#import io

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#BASE_DIR = os.path.join(BASE_DIR, '../')


def fastprint(str):
  print(str)
  sys.stdout.flush()

class MNDataset(object):
    def __init__(self, path):
        """
        :param path: path that stores preprocessed data
        """
        #self.path = os.path.join(BASE_DIR, path)
        self.path = path # absolute path of data files
        with open(self.path, "rb") as f:
        #with io.TextIOWrapper(open(self.path, "rb")) as f:
            #self.dict = pickle.load()
            self.dict = pickle.load(f)
            self.data = self.dict['data']
            self.label = self.dict['labels']
            self.graph = self.dict['graphs']

        self.len = self.data.shape[0]

    def getitem(self, idx):
        if idx>=self.len:
            idx = idx % self.len
            fastprint('reaching end of dataset, restart!')

        d, l, indptr, indices = self.data[idx,:], self.label[idx], self.graph[idx].indptr, self.graph[idx].indices
        sample = {'data': d, 'label': l, 'indptr': indptr, 'indices': indices}
        return sample

def main(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, args.path)
    dataset = MNDataset(path=args.path)
    print('total data size %d' % dataset.len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('path', type=str, help="relative path that stores data")
    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    main(args)


