import os
from datetime import datetime, timezone
from time import sleep

from embedded.args import get_embedded_listening_args
from embedded import DataCoordinator, PredictionManager, PredictionCollector, SignalUtils
from embedded.microphone import AudioBuffer
from embedded.microphone.AudioCapturer import AudioCapturer
from embedded.microphone.AudioSpectrogramStream import AudioSpectrogramStream
from embedded.microphone.SpectrogramExtractor import SpectrogramExtractor
from embedded.predictors import ModelPredictor
from embedded import SpectrogramBuffer


MODEL_INPUT_TIMESTEPS = 256
MODEL_INPUT_FREQUENCY_BINS = 77
BYTES_PER_MB = 1024*1024


def main():
    """This program requires a microphone to be connected to your computer. It will collect audio, transform it, and
            run it through the provided model, and it will do so indefinitely."""
    args = get_embedded_listening_args()

    start = datetime.now(timezone.utc)

    # This deletes files at these locations, be careful!
    os.system("rm {}".format(args.predicted_intervals_output_path))
    os.system("rm {}".format(args.blackout_intervals_output_path))

    jump = args.jump
    if jump != 0 and MODEL_INPUT_TIMESTEPS % jump != 0:
        raise ValueError("'jump' must be an even divisor of {}".format(MODEL_INPUT_TIMESTEPS))

    predictor = ModelPredictor.ModelPredictor(args.model_path, batch_size=args.batch_size, jump=jump)

    data_coordinator = DataCoordinator.DataCoordinator(
        args.predicted_intervals_output_path, args.blackout_intervals_output_path, jump=jump,
        prediction_threshold=args.prediction_threshold,
        spectrogram_capture_dir=args.spectrogram_capture_dir,
        max_captured_disk_usage=args.captured_disk_usage_limit,
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
