import os
import numpy as np
import pickle


from .jsonize_data import jsonize_all_data, OUTPUT_DIR, OUTPUT_DIR_MFCC
from .construct_dataset import create_dataset_from_json


CACHE_DIR = "cached_data"
CALLS_NAME = "calls_"
EMPTY_NAME = "empty_"


def get_dataset(name='ceb1', use_mfcc=False):
    """
    wrapper for other functions in package
    """

    outdir = OUTPUT_DIR_MFCC if use_mfcc else OUTPUT_DIR
    end = "_mfcc.p" if use_mfcc else ".p"


    fname_calls = CACHE_DIR + "/" + CALLS_NAME + name + end
    fname_empty = CACHE_DIR + "/" + EMPTY_NAME + name + end

    if os.path.isfile(fname_calls) and os.path.isfile(fname_empty):
        print("found cached dataset")
        all_calls = pickle.load(open(fname_calls, "rb"))
        all_empty = pickle.load(open(fname_empty, "rb"))
        return all_calls, all_empty 


    if not os.path.isdir(outdir): 
        raise Exception("output directory " + outdir + " does not seem to exist")

    if len(os.listdir(outdir)) == 0:
    	print("output directory " + outdir + " is empty, will jsonize data")
    	jsonize_all_data(use_mfcc=use_mfcc)

    all_calls, all_empty = create_dataset_from_json(name=name, use_mfcc=use_mfcc)
    all_calls, all_empty = map(np.array, (all_calls, all_empty))

    print("caching data...")
    pickle.dump(all_calls, open(fname_calls, "wb"), protocol=4)
    pickle.dump(all_empty, open(fname_empty, "wb"), protocol=4)

    return all_calls, all_empty 



if __name__ == '__main__':

    get_dataset()
