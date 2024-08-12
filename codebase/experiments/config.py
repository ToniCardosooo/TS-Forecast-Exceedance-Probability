from neuralforecast.losses.pytorch import SMAPE

TEST_SIZE = 3
VAL_SIZE = 1
LOG = True
MOVING_BLOCKS = True

NHITS_CONFIG = {
    'max_steps': 2000,
    'val_check_steps': 30,
    'enable_checkpointing': True,
    'start_padding_enabled': True,
    'early_stop_patience_steps': 30,
    'valid_loss': SMAPE(),
    'accelerator': 'cpu'
}

