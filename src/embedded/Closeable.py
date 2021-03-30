class Closeable:
    """Abstract base class for an object that can be closed (in response to SIGINT or SIGTERM)"""
    def __init__(self):
        pass

    def close(self):
        raise NotImplementedError()
