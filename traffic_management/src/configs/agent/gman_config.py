from configs.config import Config


class GmanConfig(Config):

    TIME_SLOT = 5
    HISTORY_STEPS = 12
    PREDICTION_STEPS = 12
    NUMBER_OF_STATT_BLOCKS = 5
    NUMBER_OF_ATTENTION_HEADS = 8
    HEAD_ATTENTION_OUTPUT_DIM = 8
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    BATCH_SIZE = 16                                     # default=32 for testing
    MAX_EPOCH = 1000
    PATIENCE = 10
    LEARNING_RATE = 0.001
    DECAY_EPOCH = 5
    LAMBDA = 100

    PREDICTED_ATTRIBUTE = "speed"                       # speed or volume
