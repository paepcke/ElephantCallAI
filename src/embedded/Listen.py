import os
from datetime import datetime, timezone
from time import sleep
import pprint
from typing import List, Callable, Tuple
import numpy as np
import torch

from embedded.Closeable import Closeable
from embedded.args import get_embedded_listening_args, check_args
from embedded import DataCoordinator, PredictionManager, PredictionCollector, SignalUtils
from embedded.led.GunshotDetectionModule import GunshotDetectionModule
from embedded.led.LedController import LedController
from embedded.microphone import AudioBuffer
from embedded.microphone.AudioCapturer import AudioCapturer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.predictors import Predictor, SingleStageModelPredictor, TwoStageModelPredictor, GunshotPredictorDecorator
from embedded import SpectrogramBuffer


MODEL_INPUT_TIMESTEPS = 256
BYTES_PER_MB = 1024*1024


def main():
    """
    This program requires a microphone to be connected to your computer. It will collect audio, transform it, and
    run it through the provided model, and it will do so indefinitely.
    """

    args = get_embedded_listening_args()
    check_args(args)

    # Pretty print all of the args used for help debugging
    if args.echo_args:
        print("Arguments to 'Listen.py':")
        pprint.pprint(vars(args))

    start = datetime.now(timezone.utc)

    # This deletes files at these locations, be careful!
    if os.path.exists(args.predicted_intervals_output_path):
        os.system(f"rm {args.predicted_intervals_output_path}")
    if os.path.exists(args.blackout_intervals_output_path):
        os.system(f"rm {args.blackout_intervals_output_path}")

    jump = args.jump
    if jump != 0 and MODEL_INPUT_TIMESTEPS % jump != 0:
        raise ValueError(f"'jump' must be an even divisor of {MODEL_INPUT_TIMESTEPS}")

    closeables: List[Closeable] = []

    predictor: Predictor
    if args.single_stage_model:
        predictor = SingleStageModelPredictor.SingleStageModelPredictor(args.model_path, batch_size=args.batch_size,
                                                                        jump=jump, half_precision=args.half_precision)
    else:
        predictor = TwoStageModelPredictor.TwoStageModelPredictor(args.model_path, batch_size=args.batch_size,
                                                                  jump=jump, half_precision=args.half_precision)
    spec_extractor = SpectrogramExtractor(nfft=args.nfft, pad_to=args.nfft, hop=args.hop,
                                          max_freq=args.max_freq, sampling_freq=args.sampling_freq)

    min_batch_size = 1 if args.allow_smaller_batches else args.batch_size
    data_coordinator = DataCoordinator.DataCoordinator(
        args.predicted_intervals_output_path, args.blackout_intervals_output_path,
        num_freq_bins=spec_extractor.num_freq_bins(),
        jump=jump,
        prediction_threshold=args.prediction_threshold,
        spectrogram_capture_dir=args.spectrogram_capture_dir,
        max_captured_disk_usage=args.captured_disk_usage_limit,
        time_delta_per_time_step=args.hop/args.sampling_freq,
        min_batch_size=min_batch_size,
        override_buffer_size=convert_mb_to_num_elements(args.spectrogram_buffer_size_mb,
                                                        spec_extractor.num_freq_bins() * SpectrogramBuffer.BYTES_PER_ELEMENT))

    audio_buffer = AudioBuffer.AudioBuffer(min_appendable_time_steps=args.nfft,
                               min_consumable_rows=args.nfft,
                               override_buffer_size=convert_mb_to_num_elements(args.audio_buffer_size_mb,
                                                                               AudioBuffer.BYTES_PER_ELEMENT))

    audio_capturer = AudioCapturer(audio_buffer, frames_per_buffer=args.nfft, sampling_freq=args.sampling_freq)

    stream_detect_func = None
    if args.gunshot_model_path is not None:
        # TODO: test both of these modes, add stuff to README about usage/physical setup of LED
        if args.led_mode:
            stream_detect_func, gunshot_detector = setup_shot_detector(args, spec_extractor, closeables)
        else:
            gunshot_model = torch.load(args.gunshot_model_path,
                                       map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            predictor = GunshotPredictorDecorator.\
                GunshotPredictorDecorator(gunshot_model, predictor, args.gunshot_detection_log_path)
            closeables.append(predictor)

    spec_stream = AudioSpectrogramStream(audio_buffer, spec_extractor,
                                         timeout=args.timeout, shot_detect_func=stream_detect_func)
    pred_mgr = PredictionManager.PredictionManager(predictor, timeout=args.timeout, verbose=args.verbose)
    pred_collector = PredictionCollector.PredictionCollector(timeout=args.timeout, verbose=args.verbose)

    closeables.extend([audio_capturer, spec_stream, pred_mgr, pred_collector, data_coordinator])
    SignalUtils.set_signal_handler(closeables, start_time=start, timeout=args.timeout)

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

    if args.gunshot_model_path is not None and args.led_mode:
        gunshot_detector.join()


def convert_mb_to_num_elements(mb: int, bytes_per_element: int) -> int:
    return mb * BYTES_PER_MB // bytes_per_element


def setup_shot_detector(args, spec_extractor: SpectrogramExtractor,
                        closeables: List[Closeable]) -> Tuple[Callable[[np.ndarray], None], GunshotDetectionModule]:
    """
    Initializes the gunshot detector in LED mode.

    :param args: The command-line arguments to this program
    :param spec_extractor: The SpectrogramExtractor object used to perform the STFT operation
    :param closeables: The running list of objects that should call 'close()' before the program terminates
    :return:
    """
    gunshot_model = torch.load(args.gunshot_model_path,
                               map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    led_controller = LedController(led_pin=args.led_pin, verbose=args.verbose)
    gunshot_detector = GunshotDetectionModule(gunshot_model, led_controller, spec_extractor)

    def detect_shots(time_domain_data):
        gunshot_detector.queue_time_domain_data(time_domain_data)

    closeables.extend([gunshot_detector, led_controller])
    return detect_shots, gunshot_detector


if __name__ == "__main__":
    main()
