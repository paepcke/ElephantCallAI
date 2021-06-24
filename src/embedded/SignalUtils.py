import sys, signal
from datetime import datetime, timezone
from typing import List

from embedded.Closeable import Closeable


def set_signal_handler(closeables: List[Closeable], start_time: datetime, timeout: bool = False):
    """
    Configures behavior on SIGTERM or SIGINT (generally a user killing the process with 'kill' or Ctrl+C).

    :param closeables: A list of objects that extend Closeable that should be closed on a signal
    :param start_time: The time the process was started
    :param timeout: whether the process is configured to time out without input
    :return:
    """

    def handler(signum, frame):
        print("Received SIGTERM or SIGINT, closing interval files and terminating prediction...", file=sys.stderr)

        for closeable in closeables:
            closeable.close()

        finish = datetime.now(timezone.utc)
        print(f"Listening process finished: Time interval was {start_time} to {finish}")
        if not timeout:
            # We don't want to wait for our threads to acquire and release any locks, this should be snappy
            sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
