import matplotlib as mpl
import os

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
import json
from functools import reduce
from multiprocessing import Pool
from typing import List, Dict
import numpy as np
np_arr_2d = np.array


from .construct_dataset import get_call_args, POOL_SIZE


CHUNK_LEN = 1024
PLOT_DIR = "plotDir"


def plot_call(arr : np_arr_2d, fname : str):
    """
    plots a figure
    """
    fig, ax = plt.subplots()
    im = plt.imshow(arr)
    plt.savefig(fname)
    plt.close()



def extract_chunks(sound_jsons : List[str]) -> List[np_arr_2d]:
    """ returns a list of all sound chunks """

    all_chunks = []
    for curr_sound_file in sound_jsons:
        arr  = json.load(open(curr_sound_file, "r"))
        arr = np.array(arr['spectrum'])
        cutoff = arr.shape[1] % CHUNK_LEN
        arr = arr[:,:-cutoff]
        num_chunks = arr.shape[1] // CHUNK_LEN
        all_chunks += np.split(arr, num_chunks, axis=1)
    return all_chunks








def plot_all_chunks(name="ceb1"):


    call_args = get_call_args(name=name)
    new_call_args = []
    for sound_files, label_file in call_args:
        new_call_args.append(sound_files)

    # for testing
    new_call_args = new_call_args[:10]

    # extract data from files
    with Pool(POOL_SIZE) as thread_pool:
        concat_func = lambda x,y : x+y
        all_chunks = reduce(concat_func , thread_pool.map(extract_chunks, new_call_args))
        chunk_args = zip(all_chunks, [PLOT_DIR + "/chunk{}.png".format(i) for i in range(len(all_chunks))])
        thread_pool.starmap(plot_call, chunk_args)







