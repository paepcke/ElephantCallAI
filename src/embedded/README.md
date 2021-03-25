# Real-time Audio Classification on an Embedded Device

This directory contains code to be used in deploying an elephant listening model (or some other kind of audio 
classification model) onto a lightweight edge computing device such as a Raspberry Pi or an NVIDIA Jetson Nano.

## Requirements

Normally, you'd just run `pip install -r requirements.txt` to download dependencies, but since you likely want to
run this on a lightweight device, binaries appropriate to your device may not be available from `pip`.

Dependencies include:

`pytorch`

`pyaudio` (install this with `$ sudo apt-get install python3-pyaudio`)

`matplotlib`

`numpy`

Your Pytorch model may have additional dependencies not specified here (such as `torchvision` or `tensorboardx`).

### Raspberry Pi 4

The Raspberry Pi 4 does not have a GPU, and it ran very slowly during testing with a 43MB ResNet model. Make sure
you are aware of performance limitations before deploying `Listen.py` on a Raspberry Pi 4.

To install PyTorch dependencies, I used `https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B`.
This may not be secure, so be mindful of your exposure. You can compile these yourself,
but it's a more involved process.

### NVIDIA Jetson Nano

NVIDIA has a forum page detailing how to get some necessary PyTorch dependencies. See 
`https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048`.

## Model Constraints

Audio is sampled at 8000 Hz and quantized to 16-bit integer samples. Your model input must be a spectrogram,
the default shape is 256 time steps by 77 frequency components. Your model output must be a 1D vector of predictions for
each time step in the input. These things can be customized, but you may have to edit the code.

Based on the configurable `jump` parameter, a number of 256x77 spectrogram frames containing a particular timestep will be used
to make predictions for that timestep. The arithmetic mean of each prediction for a particular timestep is passed into
a sigmoid function and its output is used as the model's prediction for that timestep. If this overlapping behavior is
not desired, you may pass `--jump 0` to `Listen.py`.

## Usage

To see the various command-line arguments, look at `src/embedded/args.py` or run `python3 Listen.py -h`.

To start listening to the default microphone, run `python3 Listen.py` (this file is located under `src/embedded`).

Important: Before trying to run this, make sure the `PYTHONPATH` environment variable contains the `src` directory.
`$ export PYTHONPATH=<path to this repo>/src` will accomplish this.

You must pass a path to your PyTorch model (a `.pt` file) through the `--model-path` argument.

## Output

By default, detection intervals will be saved to `/tmp/prediction_intervals.txt`, though this is customizable
through the `--predicted-intervals-output-path` argument of `Listen.py`. It contains,
in UTC time, intervals for which the model detected an elephant event.

By default, 'blackout' intervals will be saved to `/tmp/blackout_intervals.txt`, though this is customizable
through the `--blackout-intervals-output-path` argument of `Listen.py`. It contains,
in UTC time, intervals for which the model did not make a prediction. This may happen because new data is collected
faster than it can be processed. It should be unlikely unless your model is extremely slow for some reason.

If you want to save spectrogram fragments that are labeled as positive by your model, you can specify a directory to
save these fragments to. By default, fragments are not saved because this is somewhat resource-intensive. Specify
the desired directory with the `--spectrogram-capture-dir` argument to `Listen.py`. `*.npy` files will be saved there
with the detection intervals in UTC being used for the file names 
(e.g., `2021-03-25-21-57-28.671189_to_2021-03-25-21-57-30.771189.npy`). As of now, these files only contain spectrogram
data for time steps that are classified as positive.

## Contributing

Unit tests are available in `test/embedded/unit`. Integration tests (requiring a model and data of various forms) are
available in `test/embedded/integration`, but for these, outputs should be manually inspected.

## Feedback

If you have questions or comments about this module, you may contact `deschwa2@stanford.edu`.