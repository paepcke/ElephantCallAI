'''
Created on Aug 4, 2020

@author: paepcke
'''
import argparse
import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from spectrogram_dataset import SpectrogramDataset
from spectrogram_dataloader import SpectrogramDataLoader

class SpectrogramTrainer(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, snippet_db_path, input_paths):
        '''
        Constructor
        '''
        self.net = self.create_cnn()
        self.dataset = self.get_dataset(snippet_db_path, input_paths)
        
        self.dataloader = SpectrogramDataLoader(self.dataset)
        
        # Use kfold cross validation with default
        # values (5 folds, no repeat):
        self.dataloader.kfold_stratified()
        
        spectro_label_dict = next(self.dataloader)
        #*****
        print('foo')
        #*****

    #------------------------------------
    # get_dataset
    #-------------------
    
    def get_dataset(self,
                    sqlite_db_path,
                    input_paths,
                    ):
        '''
        Input paths may be a mix of files and directories
        of spectrogram snippets. Those are .pickle dataframes.
        Label files are not required, because they were 
        collected into the sqlite_db, whose path must be
        passed in. 
        
        @param input_paths:
        @type input_paths:
        @param sqlite_db_path:
        @type sqlite_db_path:
        '''
        
        #************
        #dataset = SpectrogramDataset(input_paths, sqlite_db_path)
        dataset = SpectrogramDataset(input_paths,
                                     sqlite_db_path,
                                     debugging=True
                                     )
        #************
        
        return dataset

    #------------------------------------
    # create_cnn 
    #-------------------
    
    def create_cnn(self):
        return Net()

# -------------------------- Class Net ---------------

class Net(nn.Module):
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    #------------------------------------
    # forward 
    #-------------------

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------ Main ------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Trains CNN on spectrogram snippets."
                                     )

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None)
#     parser.add_argument('dbpath',
#                         help='fully qualified path to the sqlite db that holds the snippet info',
#                         )
#     parser.add_argument('inputs',
#                         type=str,
#                         nargs='+',
#                         help='Repeatable: files and directories with spectrogram and label files')

    args = parser.parse_args();

    #*************
    args.inputs = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_0/'
    #NOTE: double dir; fix when moved dir0
    #args.inputs = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_0/Threshold_-30_MinFreq_20_MaxFreq_40_FreqCap_30_snippets_0/'
    args.dbpath = '/Users/paepcke/EclipseWorkspacesNew/ElephantCallAI/Spectrograms/Training/joint_chop_info.sqlite'
    #*************
    
    SpectrogramTrainer(args.dbpath, args.inputs)