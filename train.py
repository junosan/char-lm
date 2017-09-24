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
Script for training

Use as (for example):
    DEV="device=cuda0"
    NAME="some_name"
    MODEL_DIR=path_to_models_dir
    DATA_DIR=path_to_data_dir

    FLAGS=$DEV",floatX=float32,gpuarray.preallocate=1,base_compiledir=theano"
    THEANO_FLAGS=$FLAGS python -u train.py --data_dir=$DATA_DIR \
        --save_to=$MODEL_DIR/workspace_$NAME \
        [--load_from=$MODEL_DIR/workspace_$LOADNAME] [--seed=some_number] \
        | tee -a $MODEL_DIR/$NAME".log"

- Device "cuda$" means $-th GPU
- Flag gpuarray.preallocate reserves given ratio of GPU mem (reduce if needed)
- Flag base_compiledir directs intermediate files to pwd/theano to avoid
  lock conflicts between multiple training instances (by default ~/.theano)
- $NAME == $LOADNAME is permitted
"""

from __future__ import absolute_import, division, print_function
from six import iterkeys, itervalues, iteritems

from collections import OrderedDict
import argparse
from net import Net
from data import DataIter
import time
import numpy as np
import theano as th
import os, errno
import sys

def main():
    options = OrderedDict()

    options['input_dim']          = 27
    options['target_dim']         = 27
    options['unit_type']          = 'lstm'         # fc/lstm/gru/rhn
    options['lstm_peephole']      = True
    # options['rhn_n_layers']       = 10
    options['loss_type']          = 'crossentropy' # l2/l1/huber/crossentropy
    # options['huber_delta']        = 0.33         # depends on target's scale
    options['net_width']          = 2048
    options['net_depth']          = 1
    options['batch_size']         = 128
    options['window_size']        = 128
    options['step_size']          = 64
    options['init_scale']         = 0.02
    options['init_use_ortho']     = False
    options['weight_norm']        = False
    options['layer_norm']         = False
    options['residual_gate']      = False
    options['learn_init_states']  = False
    # options['grad_norm_clip']     = 2.       # comment out to turn off
    options['update_type']        = 'nesterov' # sgd/momentum/nesterov
    options['update_mu']          = 0.9        # for momentum/nesterov
    options['force_type']         = 'adadelta' # vanilla/adadelta/rmsprop/adam
    options['force_ms_decay']     = 0.99       # for adadelta/rmsprop
    # options['force_adam_b1']      = 0.9
    # options['force_adam_b2']      = 0.999
    options['frames_per_epoch']   = 2 * 1024 * 1024
    options['lr_init_val']        = 1e-5
    options['lr_lower_bound']     = 1e-7
    options['lr_decay_rate']      = 0.5
    options['max_retry']          = 5
    options['unroll_scan']        = False      # faster training/slower compile

    if options['unroll_scan']:
        sys.setrecursionlimit(32 * options['window_size']) # 32 is empirical

    
    """
    Parse arguments, list files, and THEANO_FLAG settings
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir' , type = str, required = True)
    parser.add_argument('--save_to'  , type = str, required = True)
    parser.add_argument('--load_from', type = str)
    parser.add_argument('--seed'     , type = int)
    args = parser.parse_args()

    # make sure directory args.save_to exists
    try:
        os.makedirs(args.save_to)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # list of context_name's (THEANO_FLAGS=contexts=... for multi GPU mode)
    c_names = [m.split('->')[0] for m in th.config.contexts.split(';')] \
              if th.config.contexts != "" else None

    # for replicating previous experiments
    seed = np.random.randint(np.iinfo(np.int32).max) \
           if args.seed is None else args.seed
    np.random.seed(seed)

    # NOTE: window_size must be the same as that given to Net
    train_data = DataIter(text_file   = args.data_dir + '/train',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          batch_size  = options['batch_size'])
    dev_data   = DataIter(text_file  = args.data_dir + '/dev',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          batch_size  = options['batch_size'])
    test_data  = DataIter(text_file  = args.data_dir + '/test',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          batch_size  = options['batch_size'])

    """
    Print summary for logging 
    """

    def print_hline(): print(''.join('-' for _ in range(79)))
    lapse_from = lambda start: ('(' + ('%.1f' % (time.time() - start)).rjust(7)
                                + ' sec)')

    print_hline() # -----------------------------------------------------------
    print('Data location : ' + args.data_dir)
    if args.load_from is not None:
        print('Re-train from : ' + args.load_from)
    print('Save model to : ' + args.save_to)

    print_hline() # -----------------------------------------------------------
    print('Options')
    maxlen = max(len(k) for k in options.keys())
    for k, v in iteritems(options):
        print('    ' + k.ljust(maxlen) + ' : ' + str(v))
    
    print_hline() # -----------------------------------------------------------
    print('Stats')
    print('    np.random.seed : ' + str(seed).rjust(10))
    print('    train set size : ' + str(train_data.size()).rjust(10))
    print('    dev   set size : ' + str(dev_data  .size()).rjust(10))
    print('    # of weights   : ', end = '')
    net = Net(options, args.save_to, args.load_from, c_names) # takes few secs
    print(str(net.n_weights()).rjust(10))


    """
    Compile th.function's (time consuming) and prepare for training 
    """

    print_hline() # -----------------------------------------------------------
    print('Compiling fwd/bwd propagators... ', end = '') # takes minutes ~ 
    start = time.time()                                  # hours (unroll_scan)
    f_fwd_bwd_propagate = net.compile_f_fwd_bwd_propagate()
    f_fwd_propagate     = net.compile_f_fwd_propagate()
    print(lapse_from(start))

    print('Compiling updater/initializer... ', end = '')
    start = time.time()
    f_update_v_params = net.compile_f_update_v_params()
    f_initialize_optimizer = net.compile_f_initialize_optimizer()
    print(lapse_from(start))


    chunk_size = options['step_size'] * options['batch_size']
    trained_frames_per_epoch = \
        (options['frames_per_epoch'] // chunk_size) * chunk_size

    def run_epoch(data_iter, lr_cur):
        """
        lr_cur sets the running mode
            float   training
            None    inference
        """
        is_training = lr_cur is not None
        if is_training:
            # apply BPTT(window_size; step_size)
            step_size = options['step_size']
        else:
            # set next_prev_idx = window_size - 1 for efficiency
            step_size = options['window_size']
        frames_per_step = step_size * options['batch_size']

        data_iter.set_step_size(step_size)

        loss_sum = 0.
        frames_seen = 0

        for input_tbi, target_tbi in data_iter:
            if is_training:
                loss = f_fwd_bwd_propagate(input_tbi, target_tbi, step_size)
            else:
                loss = f_fwd_propagate(input_tbi, target_tbi, step_size)
            
            loss_sum    += np.asscalar(loss[0])
            frames_seen += frames_per_step
            
            if is_training:
                f_update_v_params(lr_cur)
            
            if frames_seen >= trained_frames_per_epoch:
                break
        return np.float32(loss_sum / frames_seen)
    

    """
    Scheduled learning rate annealing with patience
    Adapted from https://github.com/KyuyeonHwang/Fractal
    """

    # Names for saving/loading
    name_pivot = '0'
    name_prev  = '1'
    name_best  = None # auto

    trained_frames = 0
    trained_frames_at_pivot = 0
    trained_frames_at_best = 0
    discarded_frames = 0

    loss_pivot = 0.
    loss_prev  = 0.
    loss_best  = 0.

    cur_retry = 0

    lr = options['lr_init_val']
    f_initialize_optimizer()

    net.save_to_workspace(name_prev)
    net.save_to_workspace(name_best)

    while True:
        print_hline() # -------------------------------------------------------
        print('Training...   ', end = '')
        start = time.time()
        loss_train = run_epoch(train_data, lr)
        print(lapse_from(start))

        trained_frames += trained_frames_per_epoch

        print('Evaluating... ', end = '')
        start = time.time()
        loss_cur = run_epoch(dev_data, None)
        print(lapse_from(start))

        print('Total trained frames   : ' + str(trained_frames  ).rjust(12))
        print('Total discarded frames : ' + str(discarded_frames).rjust(12))
        print('Train loss : %.6f' % loss_train)
        print('Eval  loss : %.6f' % loss_cur, end = '')

        if np.isnan(loss_cur):
            loss_cur = np.float32('inf')
        
        if loss_cur < loss_best or trained_frames == trained_frames_per_epoch:
            print(' (best)', end = '')

            trained_frames_at_best = trained_frames
            loss_best = loss_cur
            net.save_to_workspace(name_best)
        print('')

        if loss_cur > loss_prev and trained_frames > trained_frames_per_epoch:
            print_hline() # ---------------------------------------------------
            
            cur_retry += 1
            if cur_retry > options['max_retry']:
                cur_retry = 0

                lr *= options['lr_decay_rate']

                if lr < options['lr_lower_bound']:
                    break

                # cur <- pivot & prev <- cur
                discard = trained_frames - trained_frames_at_pivot
                discarded_frames += discard
                trained_frames = trained_frames_at_pivot
                net.load_from_workspace(name_pivot)
                
                f_initialize_optimizer()

                loss_prev = loss_pivot
                net.save_to_workspace(name_prev)

                print('Discard recently trained ' + str(discard) + ' frames')
                print('New learning rate : ' + str(lr))

            else:
                print('Retry count : ' + str(cur_retry) 
                      + ' / ' + str(options['max_retry']))
        else:
            cur_retry = 0

            # pivot <- prev & prev <- cur
            trained_frames_at_pivot = trained_frames - trained_frames_per_epoch

            loss_pivot, loss_prev = loss_prev, loss_cur
            name_pivot, name_prev = name_prev, name_pivot

            net.save_to_workspace(name_prev)
    

    discarded_frames += trained_frames - trained_frames_at_best
    trained_frames = trained_frames_at_best
    net.load_from_workspace(name_best)

    net.remove_from_workspace(name_pivot)
    net.remove_from_workspace(name_prev)

    print('')
    print('Best network')
    print('Total trained frames   : ' + str(trained_frames  ).rjust(12))
    print('Total discarded frames : ' + str(discarded_frames).rjust(12))
    print('[Train] loss : %.6f' % run_epoch(train_data, None))
    print('[Dev]   loss : %.6f' % run_epoch(dev_data  , None))
    print('[Test]  loss : %.6f' % run_epoch(test_data , None))
    print('')

if __name__ == '__main__':
    main()
