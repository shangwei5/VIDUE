import os
from data import adobe_videodata


class ADOBE(adobe_videodata.VIDEODATA):
    def __init__(self, args, name='GOPRO', train=True):
        super(ADOBE, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data

