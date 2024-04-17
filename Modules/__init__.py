'''
shapes of the models:
    - classification: (batch_size, in_seq_len, in_features) -> (batch_size, num_classes)
    - regression: (batch_size, in_seq_len, in_features) -> (batch_size, out_features)
    - forecasting: (batch_size, in_seq_len, in_features) -> (batch_size, out_seq_len, out_features)
    - auto-enconding: (batch_size, in_seq_len, in_features) -> (batch_size, in_seq_len, in_features)
'''

from .framework.classification_framework import *