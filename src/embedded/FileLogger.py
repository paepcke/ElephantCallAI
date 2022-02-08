from typing import IO
from datetime import datetime, timezone, timedelta

from embedded.Closeable import Closeable

# On a write, flush the file buffer if it hasn't been flushed in at least this long
FLUSH_COOLDOWN = timedelta(minutes=5)


class FileLogger(Closeable):
    """An object which exposes a simple interface to insert timestamped log data into a file."""
    log_file_path: str
    file_writer: IO
    most_recent_write: datetime

    def __init__(self, log_file_path: str):
        super().__init__()
        self.log_file_path = log_file_path
        self.file_writer = open(self.log_file_path, "a+")
        self.most_recent_flush = datetime.now(timezone.utc)

    def log(self, message: str):
        now = datetime.now(timezone.utc)
        self.file_writer.write(f"{now.isoformat()}: {message}\n")
        if now - self.most_recent_flush > FLUSH_COOLDOWN:
            self.file_writer.flush()
            self.most_recent_flush = now

    def close(self):
        self.file_writer.close()
