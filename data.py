#   Copyright 2017 Hosang Yoon
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Iterator class for time slices of sequence minibatches, where each sequence
in the minibatch starts from a random position in the text file

- Creates time slices suitable for BPTT(h; h'), where
      h = window_size, h' = step_size (see doi:10.1162/neco.1990.2.4.490)
  For training,  h = 2 h' is recommended
  For inference, h =  h'  is recommended (h > h' produces identical results,
                                          but wastes computation)
- For unroll_scan option to work in Net, window_size must be a compile time
  constant and hence cannot be changed once a Net is instantiated
- Use as
      for input_tbi, target_tbi in data_iter:
          (loop content)
  where return values are np.ndarray's of dimensions
      input     int32     [window_size][batch_size][1]
      target    int32     [window_size][batch_size][1]
  containing values 0-26 corresponding to ' ', 'a', ..., 'z'
- Unless stopped explicitly inside the loop, iterates indefinitely
- Upon each iteration, 
    - previous arrays are shifted by step_size to the left in time dimension
    - new step_size amount of data are read and put on the right
    - forward propagation is performed starting from prev_states
    - prev_states are rewound to time index (step_size - 1)
    - loss is calculated from the last step_size time indices
    - error is backpropagated on the whole window_size
- First few iterations may have zeros or irrelevant data on the left, but
  states will be reset when the real data starts and loss won't be calculated
  in or be propagated to the zero-padded/irrelevant region

Test:

from data import DataIter
data = DataIter('data/dev', 32, 16, 2)
for _ in range(8):
    i, t = data.next()
    print '[i0] ' + ''.join([chr(c + 96) for c in i[:, 0, 0]])
    print '[t0] ' + ''.join([chr(c + 96) for c in t[:, 0, 0]])
    print '[i1] ' + ''.join([chr(c + 96) for c in i[:, 1, 0]])
    print '[t1] ' + ''.join([chr(c + 96) for c in t[:, 1, 0]])
    print '     ' + '---------------||---------------'
"""

from __future__ import absolute_import, division, print_function
from six import Iterator # allow __next__ in Python 2

import numpy as np
from collections import OrderedDict

# make sure seed is set and saved before instantiating these for repeatability
class TextBatcher:
    def __init__(self, file, batch_size):
        # transform space/lowercase/uppercase to 0-26
        self._data = np.fromfile(file, dtype = '<i1') % 32
        self._n = self._data.shape[0]

        assert self._n > 0, "Empty text file"
        assert min(self._data) >= 0 and max(self._data) <= 26, "Bad text file"

        self._batch_size = batch_size
        self._idx = np.random.randint(self._n, size = batch_size)

    def next(self, step_size):
        """
        Returns tuple (input, target), where target is input 1 frame ahead
        """
        ret = np.zeros((step_size + 1, self._batch_size, 1)).astype('int32')
        next_idx = (self._idx + step_size + 1) % self._n

        for i, idx in enumerate(self._idx):
            if idx < next_idx[i]:
                ret[:, i, 0] = self._data[idx : next_idx[i]]
            else:
                d = self._n - idx
                ret[: d, i, 0] = self._data[idx :]
                ret[d :, i, 0] = self._data[: next_idx[i]]
        
        self._idx = (next_idx - 1) % self._n

        return ret[: -1], ret[1 :]

    def size(self):
        return self._n

class DataIter(Iterator):
    def __iter__(self):
        return self

    def __init__(self, text_file, window_size, step_size, batch_size):
        self._window_size = window_size
        self.set_step_size(step_size)
        
        self._data = TextBatcher(text_file, batch_size)

        # buffers for last minibatch [time][batch][class index]
        self._input  = np.zeros((window_size, batch_size, 1)).astype('int32')
        self._target = np.zeros((window_size, batch_size, 1)).astype('int32')

    def set_step_size(self, step_size):
        assert self._window_size >= step_size
        self._step_size = step_size

    def __next__(self):
        s = self._step_size

        self._input [: -s] = self._input [s :]
        self._target[: -s] = self._target[s :]

        self._input[-s :, :, :], self._target[-s :, :, :] = self._data.next(s)

        return self._input, self._target
    
    def size(self):
        return self._data.size()
