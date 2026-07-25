# config.py

BATCH_SIZE = 64

EPOCHS = 5

LEARNING_RATE = 1e-3

TRAIN_SPLIT = 0.9

# Keep the conservative Windows-compatible default.  Once the training loop
# works locally, try 2 or 4 workers to speed up image decoding.
NUM_WORKERS = 0

SEED = 42

IMAGE_SIZE = 32

NUM_CLASSES = 2

CHECKPOINT_DIR = "checkpoints"
