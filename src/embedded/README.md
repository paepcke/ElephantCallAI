# Real-time Audio Classification on an Embedded Device

This directory contains code to be used in deploying an elephant listening model (or some other kind of audio 
classification model) onto a lightweight edge computing device such as a Raspberry Pi or an NVIDIA Jetson Nano.

Sounds from a microphone are processed on the device. Spectrograms are generated from these sounds and fed as input to
a trained ML model. Time intervals of positively-classified events are logged to a file. Optionally,
spectrograms and raw predictions for time intervals containing positively-classified events can be saved to disk.

## Requirements

Normally, you'd just use `pip3` to download all of your dependencies, but since you likely want to
run this on a lightweight device, binaries appropriate to your device may not be available from `pip3`.

Dependencies include:

`sounddevice` (install this with `$ sudo apt-get install libffi-dev; pip3 install sounddevice`)

`pytorch` (methods of installing this vary based on your chosen hardware)

`matplotlib` (install this with `$ pip3 install matplotlib`)

`numpy` (install this with `$ pip3 install numpy`)

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

Note: When running the Jetson, you can use the command `$ sudo nvpmodel -m 1` to turn off two of its 4 CPU cores to save
power. This is not recommended for development with USB peripherals and monitors connected, though. Use
`$sudo nvpmodel -m 0` to restore the Jetson to full-power mode. These changes persist across reboots.

## Model Constraints

The following details constrain models to the existing codebase, but modification of the codebase
to accommodate slight differences should not be difficult.

Audio is sampled at 8000 Hz and quantized to 16-bit integer samples. Your model input must be a spectrogram,
the required shape is 256 time steps by 77 frequency components. There will also be a batch dimension;
your model must accept inputs of shape (batch_size, 256, 77). Your model output must be a 1D vector of logits for
each time step in the input. A sigmoid function will be applied to the logits and a decision function
(based on the threshold) will be applied after that. These things can be customized, but you may have to edit the code.

Currently, after taking the STFT of audio data, a transform is applied to each element x: 10*log10(x). The model should
be trained for this. We also normalize each prediction frame (256x77 input tensor) at prediction time so that
the arithmetic mean of all elements in the frame is 0 and the variance of all elements in the frame is 1. These methods
can be edited fairly easily in `FileSpectrogramStream.py`, `AudioSpectrogramStream.py`, and `PredictionUtils.py` if
necessary to accommodate models trained with different configurations.

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

### Spectrogram Capturing

If you want to save spectrogram fragments that are labeled as positive by your model, you can specify a directory to
save these fragments to. By default, fragments are not saved because this is somewhat resource-intensive. Specify
the desired directory with the `--spectrogram-capture-dir` argument to `Listen.py`. `*.pkl` files will be saved there
with the detection intervals in UTC being used for the file names 
(e.g., `2021-03-25-21-57-28.671189_to_2021-03-25-21-57-30.771189.pkl`). As of now, each of these spectrograms contains
256 time steps (about 27 seconds) of data and *every* time interval containing any predictions above the threshold
will be represented in one of these saved spectrograms.

When unpickled, these files yield Dictionary objects with two numpy arrays: one for the spectrogram and another for the
model's predictions. To find out which time steps were predicted positive, compare a time step's prediction value to the
threshold used for the invocation of `Listen.py` that generated the .pkl file. See 
`src/embedded/analysis/CapturedSpectrogramLoader.py` for an example of unpickling a file like this.

If you plan to leave `Listen.py` running for a long time, you may want to set a cap on the disk space taken up by all of
these .pkl files. Use `--captured-disk-usage-limit <NUMBER OF GIGABYTES>` to accomplish this (runs will only track 
their own disk usage, even if adding files to a directory that already has .pkl files in it, so be mindful).

## Contributing

Unit tests are available in `test/embedded/unit`. Integration tests (requiring a model and data of various forms,
not provided in this repo) are available in `test/embedded/integration`, but for these,
outputs should be manually inspected.

## Feedback

If you have questions or comments about this module, you may contact `deschwa2@stanford.edu`.