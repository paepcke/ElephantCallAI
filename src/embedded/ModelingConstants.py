# number of frequency bins in input spectrograms to current elephant models
ELEPHANT_MODEL_FREQ_BINS = 77

# number of frequency bins in input spectrograms to current gunshot models
GUNSHOT_MODEL_FREQ_BINS = 525

# max_freq arg required for SpectrogramExtractor for compatibility with current gunshot models
GUNSHOT_MODEL_MAX_FREQ_SPEC_ARG = 1024

# detection threshold for the current gunshot model
GUNSHOT_MODEL_DETECTION_THRESHOLD = 0.5

# number of time steps in one input window for the gunshot model
GUNSHOT_MODEL_TIME_STEPS = 95

# fraction of overlap between two adjacent input windows for the gunshot model
GUNSHOT_OVERLAP_FRAC = 0.25