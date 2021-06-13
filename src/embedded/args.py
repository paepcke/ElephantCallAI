import argparse


def get_embedded_listening_args():
    """Command-line arguments for Listen.py are configured here"""
    parser = argparse.ArgumentParser()

    # I/O arguments
    parser.add_argument('--predicted-intervals-output-path', type=str, default="/tmp/predicted_intervals.txt",
                        help="the location of the file that prediction intervals will be recorded to")
    parser.add_argument('--blackout-intervals-output-path', type=str, default="/tmp/blackout_intervals.txt",
                    help="the location of the file that blackout intervals will be recorded to")
    parser.add_argument('--model-path', type=str, required=True,
                        help="a path to a PyTorch model OR a directory containing two correctly-named models" +
                             " for audio classification")
    parser.add_argument('--single-stage-model', action='store_true',
                        help="Specifies that a single-stage model be used instead of a two-stage model.")
    parser.add_argument('--spectrogram-capture-dir', type=str, default=None,
                        help="A directory to save positively-labeled spectrograms to." +
                             " Spectrograms will not be saved if this argument is not specified.")
    parser.add_argument('--captured-disk-usage-limit', type=float, default=1.,
                        help="A floating-point number of gigabytes. Once this much spectrogram data has been saved to " +
                             "disk, no more spectrograms will be captured (prediction intervals will still be recorded).")
    parser.add_argument('--echo-args', action='store_true',
                        help="Causes all arguments to be printed at the very start of the process. Can be very useful " +
                             "if you're comparing different combinations of arguments.")

    # Prediction configuration arguments
    parser.add_argument('--batch-size', type=int, default=4,
                        help="How many spectrogram frames the model should be run on in parallel. More powerful " +
                             "hardware can benefit from larger values, but it may cause slowdown past a certain point.")
    parser.add_argument('--allow-smaller-batches', action='store_true',
                        help="Specify this to allow the predictor to process data as soon as there is enough for a model" +
                             "prediction instead of waiting until there is a full batch ready. This may decrease power " +
                             "efficiency but it will also decrease memory pressure on the machine. It will also allow " +
                             "predictions to be made sooner, which could be important if some real-time response is desired.")
    parser.add_argument('--jump', type=int, default=64,
                        help="The offset in time-steps of adjacent predicted spectrogram frames. Must be an even " +
                             "divisor of the model input's time dimension.")
    parser.add_argument('--prediction-threshold', type=float, default=0.5,
                        help="If a prediction is greater than or equal to this value, it will be classified as positive")
    parser.add_argument('--spectrogram-buffer-size-mb', type=int, default=16,
                        help="The size of the spectrogram buffer, in MB. This will be statically allocated. The " +
                             " optimal value will be hardware-dependent.")
    parser.add_argument('--audio-buffer-size-mb', type=int, default=16,
                        help="The size of the audio buffer, in MB. This will be statically allocated. The " +
                             " optimal value will be hardware-dependent.")

    # Debugging arguments
    parser.add_argument('--verbose', action='store_true', help="Prints information for debugging.")
    parser.add_argument('--timeout', action='store_true',
                        help="Causes the pipeline to terminate after a while if no new data is getting in. For " +
                        "debugging and testing.")

    # STFT configuration (this is model-specific): see SpectrogramExtractor.py for explanations
    parser.add_argument('--nfft', type=int, default=4096,
                        help="Window size used for creating spectrograms. " +
                             "This should match the setting used to train the model.")
    parser.add_argument('--hop', type=int, default=800,
                        help="Hop size used for creating spectrograms (hop = nfft - n_overlap). " +
                             "This should match the setting used to train the model.")
    parser.add_argument('--sampling-freq', type=int, default=8000,
                        help="The frequency at which the data is sampled, in Hz. " +
                        "This should match the setting used to train the model.")
    parser.add_argument('--max-freq', type=int, default=150,
                        help="Frequencies above this are omitted from generated spectrograms. " +
                        "This should match the setting used to train the model.")

    # TODO: add arguments that allow variation in model input shape?

    return parser.parse_args()
