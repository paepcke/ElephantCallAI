import os
from datetime import datetime, timezone
from time import sleep
import pprint

from embedded.args import get_embedded_listening_args
from embedded import DataCoordinator, PredictionManager, PredictionCollector, SignalUtils
from embedded.microphone import AudioBuffer
from embedded.microphone.AudioCapturer import AudioCapturer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.predictors import Predictor, SingleStageModelPredictor, TwoStageModelPredictor, RandomTorchvisionPredictor
from embedded import SpectrogramBuffer


MODEL_INPUT_TIMESTEPS = 256
MODEL_INPUT_FREQUENCY_BINS = 77
BYTES_PER_MB = 1024*1024


def main():
    """
    This program requires a microphone to be connected to your computer. It will collect audio, transform it, and
    run it through the provided model, and it will do so indefinitely.
    """

    args = get_embedded_listening_args()

    # Pretty print all of the args used for help debugging
    if args.echo_args:
        print("Arguments to 'Listen.py':")
        pprint.pprint(vars(args))

    start = datetime.now(timezone.utc)

    # This deletes files at these locations, be careful!
    os.system(f"rm {args.predicted_intervals_output_path}")
    os.system(f"rm {args.blackout_intervals_output_path}")

    jump = args.jump
    if jump != 0 and MODEL_INPUT_TIMESTEPS % jump != 0:
        raise ValueError(f"'jump' must be an even divisor of {MODEL_INPUT_TIMESTEPS}")

    predictor: Predictor
    if args.random_model is not None:
        predictor = RandomTorchvisionPredictor.RandomTorchvisionPredictor(model_type=args.random_model, batch_size=args.batch_size, jump=jump,
                                                                          half_precision=args.half_precision)
    elif args.single_stage_model:
        predictor = SingleStageModelPredictor.SingleStageModelPredictor(args.model_path, batch_size=args.batch_size,
                                                                        jump=jump, half_precision=args.half_precision)
    else:
        predictor = TwoStageModelPredictor.TwoStageModelPredictor(args.model_path, batch_size=args.batch_size,
                                                                  jump=jump, half_precision=args.half_precision)

    min_batch_size = 1 if args.allow_smaller_batches else args.batch_size
    data_coordinator = DataCoordinator.DataCoordinator(
        args.predicted_intervals_output_path, args.blackout_intervals_output_path, jump=jump,
        prediction_threshold=args.prediction_threshold,
        spectrogram_capture_dir=args.spectrogram_capture_dir,
        max_captured_disk_usage=args.captured_disk_usage_limit,
        time_delta_per_time_step=args.hop/args.sampling_freq,
        min_batch_size=min_batch_size,
        override_buffer_size=convert_mb_to_num_elements(args.spectrogram_buffer_size_mb,
                                                        MODEL_INPUT_FREQUENCY_BINS * SpectrogramBuffer.BYTES_PER_ELEMENT))

    audio_buffer = AudioBuffer.AudioBuffer(min_appendable_time_steps=args.nfft,
                               min_consumable_rows=args.nfft + (MODEL_INPUT_TIMESTEPS - 1) * args.hop,
                               override_buffer_size=convert_mb_to_num_elements(args.audio_buffer_size_mb,
                                                                               AudioBuffer.BYTES_PER_ELEMENT))
    spec_extractor = SpectrogramExtractor(nfft=args.nfft, pad_to=args.nfft, hop=args.hop,
                                          max_freq=args.max_freq, sampling_freq=args.sampling_freq)

    audio_capturer = AudioCapturer(audio_buffer, frames_per_buffer=args.nfft, sampling_freq=args.sampling_freq)
    spec_stream = AudioSpectrogramStream(audio_buffer, spec_extractor, timeout=args.timeout)
    pred_mgr = PredictionManager.PredictionManager(predictor, timeout=args.timeout, verbose=args.verbose)
    pred_collector = PredictionCollector.PredictionCollector(timeout=args.timeout, verbose=args.verbose)

    SignalUtils.set_signal_handler([audio_capturer, spec_stream, pred_mgr, pred_collector, data_coordinator],
                                   start_time=start, timeout=args.timeout)

    audio_capturer.start()
    if args.timeout:
        # allow time to gather initial batch of data so as not to trigger early timeouts
        sleep(27)

    spec_stream.start(data_coordinator)
    pred_mgr.start(data_coordinator)
    pred_collector.start(data_coordinator)

    spec_stream.join()
    pred_mgr.join()
    pred_collector.join()


def convert_mb_to_num_elements(mb: int, bytes_per_element: int) -> int:
    return mb * BYTES_PER_MB // bytes_per_element


if __name__ == "__main__":
    main()
