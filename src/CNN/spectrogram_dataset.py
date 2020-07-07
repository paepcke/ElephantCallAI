'''
Created on Jul 7, 2020

@author: paepcke
'''

from torch.torch.utils.data import DataSet

class SpectrogramDataset(DataSet):
    '''
    Torch dataset for iterating over spectrograms.
    '''


    def __init__(self, spectrogram_dir):
        '''
        Constructor
        '''
        