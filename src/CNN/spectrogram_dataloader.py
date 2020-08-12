'''
Created on Jul 26, 2020

@author: paepcke
'''

from contextlib import contextmanager
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SpectrogramDataloader(DataLoader):
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
        # Cache used by __len__():
        self.num_batches = None

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
        '''
        Returns number of batches that will are
        available from this loader. If drop_last
        was set to True, a possibly incomplete
        batch at the end will be dropped, and will
        therefore not be included in the length
        '''
        # Use cache if it's filled:
        if self.num_batches is not None:
            return self.num_batches
        
        num_samples = len(self.dataset)
        num_batches = num_samples / self.batch_size
        if (num_samples % self.batch_size) == 0:
            # Samples fit exactly into the desired
            # number of batches:
            self.num_batches = int(num_batches)
        
        # Will have a partially filled last batch:
        if self.drop_last:
            self.num_batches = math.floor(num_batches)
        else:
            self.num_batches = math.ceil(num_batches)

        return self.num_batches
    
    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        '''
        Returns the next item batch from
        either the validation set, or the
        test set, depending on the current-split
        switch. 
        
        Switch between train and validation set by
        calling switch_to_split(split_id), where split_id
        is {'train', 'validate'}
        
        Each item received from the underlying 
        spectrogram dataset is a dict:
           {'spectrogram' : <dataframe with spectrogram snippet>,
            'label'      : <corresponding label>
            }
            
        If drop_last was set to True when creating
        this dataloader instance, the final batch
        is discarded if it is incomplete.
        
        Returns a 2-tuple of tensor stacks, one with
        batch_size stacked spectrogram tensors, and
        one with equal-height label tensors. Each of
        the label tensors will just be a single int.
            
        @return tensor [batch_size, spectro-height, spectro_width]
        @rtype: torch.Tensor
        @raise StopIteration: when no more items
            left in queue
        '''
        batch = []
        # Assemble a list of samples into a batch:
        for _i in range(self.batch_size):
            try:
                batch.append(next(self.dataset))
            except StopIteration:
                # If we are to drop the last batch
                # if it is incomplete, do that:
                if self.drop_last or len(batch) == 0:
                    raise StopIteration()
                else:
                    # Return the final partial batch:
                    return self.to_batch_d_tensor(batch)
                
        # Successfully filled a complete batch:
        return self.to_batch_d_tensor(batch)

    #------------------------------------
    # to_batch_d_tensor 
    #-------------------
    
    def to_batch_d_tensor(self, spectro_label_dict_list):
        '''
        Takes a list of dicts, and returns
        two multi-d tensors. Each dict is 
        {'spectrogram' : pd.Dataframe,
         'label'       : int}
         
        Turn each df into a tensor, and stack
        those into batch-size planes.
        
        @param spectro_label_dict_list:
        @type spectro_label_dict_list:
        @return one batch_size-D tensor of 
            spectrograms, and one batch_size-D
            tensor of label ints
        '''
        res_spectro_tns = []
        res_label_tns   = []
        for spectro_label in spectro_label_dict_list:
            spectro_tns = torch.tensor(spectro_label['spectrogram'].values)
            label_tns   = torch.tensor(spectro_label['label'])
            
            (data_height, data_width) = spectro_tns.shape
            
            # The unsqueeze() will be collapsed back
            # down by torch.cat() below:
            res_spectro_tns.append(spectro_tns.unsqueeze(0))
            res_label_tns.append(label_tns.unsqueeze(0))

        res_spectros = torch.cat(res_spectro_tns)
        res_labels   = torch.cat(res_label_tns)
        
        # Now have res_spectros.shape:
        #    [2,HEIGHT, WIDTH)
        # The unsqueeze adds an empty dimenasion
        # to indicate single-channel to make
        #    [1,2,HEIGHT,WIDTH] 
        res_spectros = res_spectros.unsqueeze(0)
        res_labels   = res_labels.unsqueeze(0)
        
        # But ResNet wants:
        #    [batch_size, channels, HEIGHT, WIDTH]
        
        res_spectros_batchsize_chnls_h_w = \
            res_spectros.reshape([self.batch_size,
                                  1,   # Single channel
                                  data_height,
                                  data_width
                                  ])
        # Make labels stacked like the samples:
        res_labels = res_labels.reshape(self.batch_size, 1)

        return (res_spectros_batchsize_chnls_h_w,
                res_labels)

    #------------------------------------
    # __iter__ 
    #-------------------

    def __iter__(self):
        return self
    
    #------------------------------------
    # __getitem__
    #-------------------

    def __getitem__(self, indx):
        return self.dataset[indx]

# -------------------- Multiprocessing Dataloader -----------

class MultiprocessingDataloader(SpectrogramDataloader):
    
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

