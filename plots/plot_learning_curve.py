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

from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
import re

def re_num(s):
    r = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', s)
    return float(r[0])

def main():

    name = '2x1024'

    lc = np.loadtxt(name + '.csv', delimiter=',')[: -1]
    
    lr_boundaries = []

    cumul_t = 0
    with open('../models/' + name + '.log') as f:
        lines = [l.rstrip('\n') for l in f]
        for i, line in enumerate(lines):
            if 'Training...   (' in line:
                cumul_t += re_num(lines[i]) + re_num(lines[i + 1])
            if 'New learning rate :' in line:
                lr_boundaries.append(cumul_t)

    lr_boundaries = [t / 3600. for t in lr_boundaries] # in hours

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    figsize = [9, 3.5]

    # 
    fig = plt.figure(figsize=figsize)
    plt.get_current_fig_manager().window.wm_geometry("+0+0")

    plt.subplot(1, 2, 1)

    h = plt.plot(lc[:, 0] / 3600., lc[:, 1] / np.log(2), '--', # training
                 lc[:, 0] / 3600., lc[:, 2] / np.log(2), '-')  # validation

    h[1].set_color(h[0].get_color())
    plt.setp(h[0], linewidth=1)

    plt.ylim([1.3, 2.9])

    plt.xlabel('Training time (hour)')
    plt.ylabel('BPC')

    plt.legend(h, ['training', 'validation'])

    plt.gca().set_xticks(lr_boundaries, minor=True)
    plt.gca().xaxis.grid(True, which='minor')
    
    # turn off minor tick lines
    for tic in plt.gca().xaxis.get_minor_ticks():
        tic.tick1On = tic.tick2On = False

    # zoomed in
    plt.subplot(1, 2, 2)

    h = plt.plot(lc[:, 0] / 3600., lc[:, 1] / np.log(2), '--', # training
                 lc[:, 0] / 3600., lc[:, 2] / np.log(2), '-')  # validation

    h[1].set_color(h[0].get_color())
    plt.setp(h[0], linewidth=1)

    plt.xlim([2.5, 7.0])
    plt.ylim([1.3, 1.8])

    plt.xlabel('Training time (hour)')
    plt.ylabel('BPC')

    plt.legend(h, ['training', 'validation'])

    plt.gca().set_xticks(lr_boundaries, minor=True)
    plt.gca().xaxis.grid(True, which='minor')
    
    # turn off minor tick lines
    for tic in plt.gca().xaxis.get_minor_ticks():
        tic.tick1On = tic.tick2On = False
    

    plt.tight_layout()
    fig.savefig('learning_curve.eps', format='eps')

    plt.get_current_fig_manager().window.attributes('-topmost', 1)
    plt.show()

if __name__ == '__main__':
    main()
