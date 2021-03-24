from typing import IO, Tuple
from datetime import datetime, timezone, timedelta

from embedded.Closeable import Closeable

# On a write, flush the file buffer if it hasn't been flushed in at least this long
FLUSH_COOLDOWN = timedelta(minutes=5)


class IntervalRecorder(Closeable):
    interval_file_path: str
    file_writer: IO
    most_recent_write: datetime

    def __init__(self, interval_file_path: str):
        super().__init__()
        self.interval_file_path = interval_file_path
        self.file_writer = open(self.interval_file_path, "a+")
        self.most_recent_flush = datetime.now(timezone.utc)

    def write_interval(self, interval: Tuple[datetime, datetime]):
        self.file_writer.write("{},{}\n".format(interval[0].isoformat(), interval[1].isoformat()))
        now = datetime.now(timezone.utc)
        if now - self.most_recent_flush > FLUSH_COOLDOWN:
            self.file_writer.flush()
            self.most_recent_flush = now

    def close(self):
        self.file_writer.close()
