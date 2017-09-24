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

def main():

    name_lists = []
    legend_lists = []
    fignames = []
    ylims = []

    # 4x512 and variants
    name_lists.append(['4x512',
                       '4x512_wn',
                       '4x512_rg',
                       '4x512_ln',
                       '4x591_gru'])
    legend_lists.append(['4x512 LSTM',
                         '4x512 LSTM WN',
                         '4x512 LSTM RG',
                         '4x512 LSTM LN',
                         '4x591 GRU'])
    fignames.append('exp_arch_lc_4x512')
    ylims.append([1.45, 1.85])

    # 3x1024 and variants
    name_lists.append(['3x1024',
                       '3x1024_wn',
                       '3x1024_rg'])
    legend_lists.append(['3x1024 LSTM',
                         '3x1024 LSTM WN',
                         '3x1024 LSTM RG'])
    fignames.append('exp_arch_lc_3x1024')
    ylims.append([1.45, 1.85])

    # 1x2048 vs 4x1024
    name_lists.append(['1x2700',
                       '4x1024_wn_rg',
                       '1x1024_rhn'])
    legend_lists.append(['1x2700 LSTM',
                         '4x1024 LSTM WN RG',
                         '1x1024 10-layer RHN'])
    fignames.append('exp_arch_lc_final')
    ylims.append([1.3, 1.7])
    

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    figsize = [7, 3.5]

    for names, legend, figname, ylim, i in \
        zip(name_lists, legend_lists, fignames, ylims, range(len(ylims))):
        lc = {} # learning curve data

        for name in names:
            data = np.loadtxt(name + '.csv', delimiter=',')
            lc[name] = data[: -1]

        # 
        fig = plt.figure(figsize=figsize)
        plt.get_current_fig_manager().window.wm_geometry("+0+%d" % (350 * i))

        l_va = []
        for name in names:
            l_va += [lc[name][:, 0] / 3600., lc[name][:, 2] / np.log(2), '-']
        
        h_va = plt.plot(*l_va)

        plt.ylim(ylim)

        plt.xlabel('Training time (hour)')
        plt.ylabel('Validation set BPC')

        plt.legend(h_va, legend)

        plt.tight_layout()
        fig.savefig(figname + '.eps', format='eps')

    plt.get_current_fig_manager().window.attributes('-topmost', 1)
    plt.show()

if __name__ == '__main__':
    main()
