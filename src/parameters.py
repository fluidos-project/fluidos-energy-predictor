# Dataset parameters
# Span of a single step in minutes
from src.support.dt import WEEK_IN_MINUTES

GRANULARITY = 15
# Offset in minutes between each file
# OFFSET = 24 * 60

STEPS_IN = WEEK_IN_MINUTES // GRANULARITY
STEPS_OUT = 1
N_FEATURES = 2
SPLIT = 0.25

FILTERS = 144
KSIZE = 3

OVERHEAD = 1

PATIENCE = 150

LEARNING_RATE = 0.02

LOG_FOLDER = "out"

DEFAULT_MODEL = "model1"
MODEL_FOLDER = "models"
GCD_FOLDER = "data/gcd"
SPEC_FOLDER = "data/spec2008_agg"
CACHE_FOLDER = "data/cache"
BANLIST_FILE = "banlist"

TEST_FILE_AMOUNT = 24
TRAIN_FILE_AMOUNT = 24


