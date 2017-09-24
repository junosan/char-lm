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

def get_n_weights(file_name):
    with open(file_name) as f:
        for line in [l.rstrip('\n') for l in f]:
            if '    # of weights   :' in line:
                return int(re.findall('\d+', line)[0])

def main():

    depths = [1, 2, 3, 4]
    widths = [128, 256, 512, 1024]

    # key (depth, width)
    lc    = {} # learning curve data
    train = {} # final train set loss
    valid = {} # final validation set loss
    test  = {} # final test set loss
    weigh = {} # number of weights

    for d in depths:
        for w in widths:
            data = np.loadtxt(str(d) + 'x' + str(w) + '.csv',
                              delimiter=',')
            lc   [(d, w)] = data[: -1]
            train[(d, w)] = data[-1, 1]
            valid[(d, w)] = data[-1, 2]
            test [(d, w)] = data[-1, 0]
            weigh[(d, w)] = get_n_weights('../models/' + str(d) +
                                          'x' + str(w) + '.log')

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    figsize = [9, 7]

    # loss vs complexity
    fig = plt.figure(figsize=figsize)
    plt.get_current_fig_manager().window.wm_geometry("+0+0")
    
    plt.subplot(1, 2, 1)
    x = np.log2(widths)
    y_tr = [[train[(d, w)] / np.log(2) for w in widths] for d in depths]
    y_te = [[test [(d, w)] / np.log(2) for w in widths] for d in depths]
    
    h_te = plt.plot(x, y_te[0], 'o-' , x, y_te[1], 's-' ,
                    x, y_te[2], '^-' , x, y_te[3], 'v-' )
    # h_tr = plt.plot(x, y_tr[0], 'o--', x, y_tr[1], 's--',
    #                 x, y_tr[2], '^--', x, y_tr[3], 'v--')

    # colors = [h.get_color() for h in h_te]
    # for h, c in zip(h_tr, colors):
    #     h.set_color(c)

    # plt.setp(h_tr, linewidth=1)

    plt.ylim([1.5, 2.15])
    
    plt.gca().set_yticks([(n + 0.5) / 10.0 for n in range(15, 22)], minor=True)
    plt.grid(True, which='both', axis='y')

    plt.xlabel('Network width')
    plt.ylabel('Test set BPC')

    plt.xticks(x, [str(w) for w in widths])
    plt.legend(h_te, ['Depth ' + str(d) for d in depths])


    # number of weights vs complexity
    plt.subplot(2, 2, 2)

    y_nw = [[weigh[(d, w)] for w in widths] for d in depths]

    h_nw = plt.semilogy(x, y_nw[0], 'o-', x, y_nw[1], 's-',
                        x, y_nw[2], '^-', x, y_nw[3], 'v-')

    plt.xlabel('Network width')
    plt.ylabel('Number of weight parameters')

    plt.xticks(x, [str(w) for w in widths])
    plt.legend(h_nw, ['Depth ' + str(d) for d in depths])


    # training time vs complexity
    plt.subplot(2, 2, 4)
    y_ti = [[lc[(d, w)][-1, 0] / 3600. for w in widths] for d in depths]

    h_ti = plt.plot(x, y_ti[0], 'o-', x, y_ti[1], 's-',
                    x, y_ti[2], '^-', x, y_ti[3], 'v-')

    plt.xlabel('Network width')
    plt.ylabel('Total training time (hour)')

    plt.xticks(x, [str(w) for w in widths])
    plt.legend(h_ti, ['Depth ' + str(d) for d in depths])

    plt.tight_layout()
    fig.savefig('complexity.eps', format='eps')

    plt.get_current_fig_manager().window.attributes('-topmost', 1)
    plt.show()

if __name__ == '__main__':
    main()
