# src/config.py

# === data paths ===
CLASS_DIR = "data/classification"
GEN_FILE = "data/generation/raw_text"  # Will automatically check for .csv extension

# === tokenization ===
CLS_MAX_LEN = 256    # Increased for better context
GEN_MAX_LEN = 100    # Max length for generation

# === model/training params for classification ===
EMB_DIM = 100        # GloVe 6B.100d
HIDDEN_DIM = 256     # Balanced size
NUM_LAYERS = 3       # More layers for better representation
DROPOUT = 0.3        # Higher dropout for regularization
BIDIRECTIONAL = True

# Training hyperparameters
BATCH_SIZE = 64      # Smaller batch size for better gradients
LR = 1e-3            # Higher initial learning rate
WEIGHT_DECAY = 1e-5  # Regularization
EPOCHS = 30          # More epochs with early stopping

SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# === generation model params ===
# Use 100d to match GloVe embeddings
GEN_EMB_DIM = 100    # Changed from 768 to match GloVe
GEN_HIDDEN_DIM = 512 # Reasonable hidden size
GEN_NUM_LAYERS = 2
GEN_DROPOUT = 0.3
GEN_BATCH_SIZE = 32
GEN_LR = 1e-3
GEN_EPOCHS = 20      # More epochs for generation task