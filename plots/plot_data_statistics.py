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
from collections import Counter

def main():

    train = np.fromfile('../data/train', dtype = '<i1') % 32
    valid = np.fromfile('../data/dev'  , dtype = '<i1') % 32
    test  = np.fromfile('../data/test' , dtype = '<i1') % 32

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[4.5 * 3, 3.5])
    plt.get_current_fig_manager().window.wm_geometry("+0+0")

    for i, data, name in zip(range(3), [train, valid, test],
                                       ['Training', 'Validation', 'Test']):
        freq_dict = Counter(data)
        
        x = range(27)
        
        freq = [freq_dict[k] for k in x]
        tot = sum(freq)
        
        y = [float(f) / tot for f in freq]

        entropy = sum([-p * np.log2(p) for p in y])

        plt.subplot(1, 3, i + 1)
        plt.bar(x, y)

        plt.ylim([0.0, 0.175])

        plt.ylabel('Occurrence probability')
        plt.xticks(x, [chr(i + 96) if i > 0 else '_' for i in x])
        plt.title(name + ' set')

        print(name.ljust(len('Validation')) + ' set: %.3f bpc' % entropy)

    plt.tight_layout()
    fig.savefig('char_probability.eps', format='eps')
    plt.get_current_fig_manager().window.attributes('-topmost', 1)
    plt.show()

if __name__ == '__main__':
    main()
