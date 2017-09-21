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
Script for generating learning curve plots from log files

Use as:
    python plot_log.py path/to/log.log                  (generates a plot)
    python plot_log.py path/to/log.log path/to/data.csv (saves plot data)
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import argparse
import matplotlib
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs = '*', type = str)
    args = parser.parse_args()

    if len(args.args) == 0 or len(args.args) > 2:
        print("Invalid arguments")
        return

    log_file = args.args[0]
    data_file = None if len(args.args) == 1 else args.args[1]

    def re_num(str):
        r = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',
                       str)
        return float(r[0])
    
    times = [] # cumulative time from start of training
    train = [] # training losses
    valid = [] # validation losses

    final_train = 0.0
    final_valid = 0.0
    final_test  = 0.0

    cumul_t = 0
    with open(log_file) as f:
        lines = [l.rstrip('\n') for l in f]
        for i, line in enumerate(lines):
            if 'Training...   (' in line:
                cumul_t += re_num(lines[i]) + re_num(lines[i + 1])
                times.append(cumul_t)
                train.append(re_num(lines[i + 4]))
                valid.append(re_num(lines[i + 5]))
            
            if '[Train] loss :' in line:
                final_train = re_num(lines[i + 0])
                final_valid = re_num(lines[i + 1])
                final_test  = re_num(lines[i + 2])
    
    if data_file is not None:
        # except last: cumulative time, train loss, validation loss
        # last line  : test loss      , train loss, validation loss
        times.append(final_test)
        train.append(final_train)
        valid.append(final_valid)

    times = np.array(times)
    train = np.array(train)
    valid = np.array(valid)

    if data_file is None:
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.figure(figsize = [10, 4])
        
        plt.subplot(1, 2, 1)
        idx = 1 + np.array(range(times.shape[0]))
        handles_1 = plt.plot(idx, train, idx, valid)
        plt.xlabel('Number of epochs')
        plt.ylabel('Average cross entropy')
        plt.legend(handles_1, ['training', 'validation'])

        plt.subplot(1, 2, 2)
        handles_2 = plt.plot(times, train, times, valid)
        plt.xlabel('Training time (sec)')
        plt.ylabel('Average cross entropy')
        plt.legend(handles_2, ['training', 'validation'])

        cfm = plt.get_current_fig_manager() 
        cfm.window.attributes('-topmost', 1)

        plt.show()
        
    else:
        arr = np.transpose(np.array([times, train, valid]))
        np.savetxt(data_file, arr, delimiter = ',')

if __name__ == '__main__':
    main()
