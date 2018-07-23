"""
settings.py - Global settings for all nets
author : Benjamin Blundell
email : me@benjamin.computer
"""

class NeuralGlobals (object):
  ''' Master set of global settings.'''
  def __init__(self):
    from . import acids
    self.num_acids = acids.NUM_ACIDS
    self.max_cdr_length = 28 # TODO - We should set this from the DB data
    self.batch_size = 20
    self.window_size = 4
    self.num_epochs = 5
    self.pickle_filename = 'data.pickle'
    self.save_name = "model.ckpt"
    self.save_path = "./saved/"
    self.learning_rate = 0.1
    self.device = '/gpu:0'
    self.error_window = 10
    self.error_delta = 0.05
    self.absolute_error = 0.02
    self.decay_learning = True
    self.type_in = None
    self.type_out = None
    self.decay_rate = 0.98
    self.dropout = 0.8
    self.datasize = 10 # Number of files that makeup our train set
    self.lstm_size = 256 # number of neurons per LSTM cell do we think? 
