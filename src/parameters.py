# Dataset parameters
# Span of a single step in minutes
import os

from support.dt import WEEK_IN_MINUTES

GRANULARITY = 15
# Offset in minutes between each file
# OFFSET = 24 * 60

# how many minutes are there in a week, divided by 'how often'
STEPS_IN = WEEK_IN_MINUTES // GRANULARITY
STEPS_OUT = (4, 2)
N_FEATURES = 2
SPLIT = 0.25

USIZE = 20

OVERHEAD = 1

PATIENCE = 20

LEARNING_RATE = 0.002

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
