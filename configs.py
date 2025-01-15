import os
from datetime import datetime
import tensorflow as tf

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/09_translation_transformer",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        )
        self.num_layers = 4
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.dropout_rate = 0.1
        self.batch_size = 16
        self.train_epochs = 5
        # CustomSchedule parameters
        self.init_lr = 0.00001
        self.lr_after_warmup = 0.0005
        self.final_lr = 0.0001
        self.warmup_epochs = 2
        self.decay_epochs = 18

        # Additional configurations you might need
        self.steps_per_epoch = 100  # Example: Set this based on your dataset size
        self.total_steps = self.train_epochs * self.steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch

        # Learning rate schedule using Cosine Decay
        self.learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=self.total_steps,
            alpha=0.0,  # Minimum learning rate after decay
            warmup_target=self.lr_after_warmup,  # Target learning rate after warmup phase
            warmup_steps=self.warmup_steps  # Steps for the warmup phase
        )
        
        # The rest of the configuration
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)