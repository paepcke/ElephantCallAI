'''
Created on Jul 26, 2020

@author: paepcke
'''
from contextlib import contextmanager

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SpectrogramDataLoader(DataLoader):
    '''
    A dataloader that works with instances of 
    SpectrogramDataset (see spectrogram_dataset.py).
    
    Samples are delivered as dicts, and may be
    obtained by either of two methods:
    
        for sample_dict in my_dataloader:
            print(sample_dict['spectrogram']
            print(sample_dict['label']
            
    and:
        sample_dict_14 = my_dataloader[14]
    
    
    Best used with this module's cross validation
    facility. See kfold() and kfold_stratified().
    The latter method ensures class balance in each
    fold. See corresponding sklearn methods, which
    are used under the cover.
    
    For distributed training, use the MultiprocessingDataloader
    subclass.
    
    This class simply wraps spectrogram_dataset instance.
    See header comment in spectrogram_dataset.py for lots more
    information, such as switching between 
    test and validation folds.
    
    Instances of this class can be used like any other Pytorch
    dataloader.
    
    This class adds a dataset split context manager.
    It allows callers to interact temporarily with 
    a particular split: test/validate, and
    then return to the current split. Example:
    
      with set_split_id('validate'):
          avg_val_accuracy = total_eval_accuracy / len(dataloader)
    
    '''
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    #------------------------------------
    # kfold 
    #-------------------
    
    def kfold(self,
              n_splits=5,
              n_repeats=0,
              shuffle=False,
              random_state=None):
        '''
        See header comment at kfold() in spectrogram_dataset.py
        for more detail.
        
        Partitions the underlying dataset into 
        n_splits folds. All subsequent calls to 
        next() return data items either from the training
        set or the validation set, depending on the 
        current split setting ('train' or 'validate'). 
        
        Each fold 80% of the data in a training set of n-1
        samples, and a validation set containing 10%. As
        method next() is continue to be called, folds are
        automatically turned over to produce different
        train/validate mixes.
        
        The entire process can be repeated any number of
        times, resulting in n-times k-fold cross validation. 
        
        See also kfold_stratified() 
         
        @param n_splits: number of desired splits or folds.
            Equivalent to the 'k' in k-fold cross validation
        @type n_splits: int
        @param n_repeats: number of times the entire fold
            procedure is repeated
        @type n_repeats: int 
        @param shuffle: whether or not to shuffle the 
            folds (no shuffling inside folds themselves).
        @type shuffle: bool
        @param random_state: if set, guarantees repeatable
            operation when shuffle is True. No meaning
            when shuffle is False.
        @type random_state: bool
        @return: nothing
        '''
        
        self.dataset.kfold(n_splits=n_splits,
                           n_repeats=n_repeats,
                           shuffle=shuffle,
                           random_state=random_state)

    #------------------------------------
    # kfold_stratified 
    #-------------------

    def kfold_stratified(self, 
               n_splits=5,
               n_repeats=0,
               shuffle=False,
               random_state=None
               ):
        '''
        Uses sklearn's StratifiedKFold and RepeatedStratifiedKFold facility.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

        Like kfold(), but all folds are balanced to contain roughly
        equal numbers of each label class.

        See also method kfold() for non-balanced folding.
    
        More detail in spectrogram_dataset.py's kfold_stratified()
        method.
        
        After calling this method, calls to next() will
        return train samples. I.e. the current queue is set
        to self.train_queue. Switch to validation set using 
        switch_to_split().
        
        @param n_splits: number of folds to create 
        @type n_splits: int
        @param n_repeats: number times fold splitting should
            be repeated (n-times k-fold cross validation.
            Set to zero, the method uses sklearn KFold class,
            else it uses the sklearn.RepeatedKFold
        @type n_repeats: int
        @param shuffle: whether or not to shuffle the 
            data before splitting. Once split, the 
            data in the folds are not shuffled 
        @type shuffle: bool
        @param random_state: if shuffle is set to True,
            this argument allows for repeatability over
            multiple runs
        @type random_state: int
        @returns nothing
        '''
        
        self.dataset.kfold_stratified(
                     n_splits=n_splits,
                     n_repeats=n_repeats,
                     shuffle=shuffle,
                     random_state=random_state)

    #------------------------------------
    # get_n_splits 
    #-------------------
    
    def get_n_splits(self):
        '''
        Returns the number of splitting iterations in the cross-validator,
        i.e. number of folds. Takes into account repetitions.
        '''
        return self.dataset.get_n_splits()

    #------------------------------------
    # split_id 
    #-------------------
    
    def split_id(self):
        '''
        Return this loader's dataset split id: 'train',
        'validate', or 'test'. Change split_id
        with 
        '''
        return self.dataset.curr_split_id()

    #------------------------------------
    # switch_to_split 
    #-------------------
    
    def switch_to_split(self, split_id):
        '''
        Flip the underlying dataset to feed
        out either training samples, or validation
        samples. Makes sense to call only after 
        calling kfold() or kfold_stratified() first.
         
        @param split_id: desired source of samples
        @type split_id: {'train', 'validate'}
        '''
        self.dataset.switch_to_split(split_id)

    #------------------------------------
    # __len__ 
    #-------------------

    def __len__(self):
        return len(self.dataset)
    
    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        '''
        Returns the next item from
        either the validation set, or the
        test set, depending on the current-split
        switch. 
        
        Each item is a dict:
           {'snippet_df' : <dataframe with spectrogram snippet>,
            'label'      : <corresponding label>
            }
            
        Switch between train and validation set by
        calling switch_to_split(split_id), where split_id
        is {'train', 'validate'} 
            
        @return a spectgrogram and associated label
        @rtype: {str : pd.dataframe, str : int}
        @raise StopIteration: when no more items
            left in queue
        '''
        
        return next(self.dataset)

    
    #------------------------------------
    # __getitem__
    #-------------------

    def __getitem__(self, indx):
        return self.dataset[indx]

# -------------------- Multiprocessing Dataloader -----------

class MultiprocessingDataloader(SpectrogramDataLoader):
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, dataset, world_size, node_rank, **kwargs):
        
        self.dataset  = dataset
        
        self.sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=node_rank
                )

        super().__init__(dataset,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True,
                         sampler=self.sampler,
                         **kwargs)

    #------------------------------------
    # set_epoch 
    #-------------------

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

# ------------------------ set_split_id Context Manager



#------------------------------------
# set_split_id [Context manager] 
#-------------------
    
@contextmanager
def set_split_id(dataloader, tmp_split_id):
    '''
    Allows temporary setting of split_id like this:
    
      with set_split_id(dataloader, 'validate'):
          dataloader.reset_split()
          
    or: get the validate split's length:

      with set_split_id('validate'):
          avg_val_accuracy = total_eval_accuracy / len(dataloader)
          
    The above temporarily sets the dataloader's split
    to 'validate' for the duration of the 'with' body.
    Then the split is returned to the original value.
    
    @param dataloader: dataloader whose split is to be 
        temporarily changed. 
    @type dataloader: BertFeederDataloader
    @param tmp_split_id: the split id to which the dataloader
        is to be set for the scope of the with statement
    @type tmp_split_id: {'train'|'validate'|'test'}
    '''
    saved_split_id = dataloader.curr_split()
    dataloader.switch_to_split(tmp_split_id)
    try:
        yield
    finally:
        dataloader.switch_to_split(saved_split_id)

