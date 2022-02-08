class AudioBufferMetadata:
    unprocessed_end: int
    allocated_begin: int
    rows_allocated: int

    def __init__(self, buffer):
        self.unprocessed_end = buffer.unprocessed_end
        self.allocated_begin = buffer.allocated_begin
        self.rows_allocated = buffer.rows_allocated
