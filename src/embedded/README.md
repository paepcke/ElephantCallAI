# Real-time Audio Classification

This directory contains code to be used in deploying an elephant listening model (or some other kind of model) onto a
 lightweight edge computing device such as a Raspberry Pi or an NVIDIA Jetson Nano.
 
## Requirements

Normally, you'd just run `pip install -r requirements.txt` to download dependencies, but since you likely want to
run this on a lightweight device, binaries appropriate to your device may not be available from `pip`.

Dependencies include:

`pytorch`

`pyaudio`

`matplotlib`

`numpy`

Your Pytorch model may have additional dependencies not specified here.

### Raspberry Pi 4

This section is under construction

### NVIDIA Jetson Nano

This section is under construction

## Model Constraints

Audio is sampled at 8000 Hz and quantized to 16-bit integer samples. Your model input must be a spectrogram,
the default shape is 256 time steps by 77 frequency components. Your model output must be a 1D vector of predictions for
each time step in the input. These things can be customized, but you may have to edit the code.

## Usage

A scheme for accepting command-line arguments is not yet implemented.

To start listening to the default microphone, run `python EndToEndTest.py` (this file is located under `test/embedded/microphone`). For now, there are constant variables
you may want to edit to customize things like your model and the output location.

Important: Before trying to run this, make sure the `PYTHONPATH` environment variable contains the `src` directory.
`$ export PYTHONPATH=<path to this repo>/src` will accomplish this.

## Output

By default, detection intervals will be saved to /tmp/prediction_intervals.txt, though this is customizable. It contains,
in UTC time, intervals for which the model detected an elephant event.
By default, 'blackout' intervals will be saved to /tmp/blackout_intervals.txt, though this is customizable. It contains,
in UTC time, intervals for which the model did not make a prediction. This may happen because new data is collected
faster than it can be processed. It should be unlikely unless your model is extremely slow for some reason.

## Feedback

If you have questions or comments about this module, you may contact `deschwa2@stanford.edu`.