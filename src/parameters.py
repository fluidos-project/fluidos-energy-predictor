# Dataset parameters
# Span of a single step in minutes
from support.dt import WEEK_IN_MINUTES
import os

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

LOG_FOLDER = os.environ.get("OUT_TRAINING_FOLDER", "out")
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "models")
DATA_FOLDER = os.environ.get("DATA_FOLDER", "data")

GCD_FOLDER = os.path.join(DATA_FOLDER, "gcd")
SPEC_FOLDER = os.path.join(DATA_FOLDER, "spec2008_agg")
CACHE_FOLDER = os.path.join(DATA_FOLDER, "cache")

DEFAULT_MODEL = "model1"
BANLIST_FILE = "banlist"

TEST_FILE_AMOUNT = 24
TRAIN_FILE_AMOUNT = 24
