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
Script for generating text from models trained by train.py
Set model path and number of characters to be generated in gen_text.cfg

Use as:
    python gen_text.py 'some initial text to initialize the states of RNNs'

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import argparse
from net import Net
from collections import OrderedDict

def main():
    # clean input text
    parser = argparse.ArgumentParser()
    parser.add_argument('text' , type = str)
    args = parser.parse_args()

    text = str(args.text).lower()
    allowed = ' abcdefghijklmnopqrstuvwxyz'
    for c in text:
        if c not in allowed:
            print("Only characters in '" + allowed + "' are allowed")
            return

    # read settings
    model = None
    n_chars = None
    with open('gen_text.cfg') as f:
        for line in [l.rstrip('\n') for l in f]:
            if 'MODEL' in line:
                model = line[line.find('=') + 1 :]
            if 'CHARS' in line:
                n_chars = int(line[line.find('=') + 1 :])
    
    # initialize RNN
    options = OrderedDict()
    options['step_size']  = 1
    options['batch_size'] = 1

    net = Net(options, None, model)
    f_fwd_propagate = net.compile_f_fwd_propagate()

    itext = [ord(c) % 32 for c in text]
    for i in itext:
        pred = f_fwd_propagate(np.int32(i).reshape((1,1,1)))

    print(text, end = '')

    # generate text
    for _ in range(n_chars):
        # using float32 as is leads to numpy throwing due to
        # probability sum sometimes exceeding 1.0
        p = np.float64(pred[0].flatten())

        # multinomial requires sum(p[: -1]) <= 1.0
        p[: -1] = p[: -1] / np.sum(p[: -1]) * (1 - p[-1])
        i = np.random.multinomial(1, p).argmax()

        print(chr(i + 96) if i > 0 else ' ', end = '')
        pred = f_fwd_propagate(np.int32(i).reshape((1,1,1)))

    print('')

if __name__ == '__main__':
    main()
