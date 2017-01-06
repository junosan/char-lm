#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Script for training

Use as:
    THEANO_FLAGS=floatX=float32,device=$DEV,gpuarray.preallocate=1 python -u \
    train.py --data_dir=$DATA_DIR --save_to=$WORKSPACE_DIR/workspace_$NAME \
    [--load_from=$WORKSPACE_DIR/workspace_$LOAD] \
    | tee -a $WORKSPACE_DIR/$NAME".log"
where $DEV=cuda0, etc (it is ok to set $NAME == $LOAD)
"""

from __future__ import print_function # for end option in print()
from collections import OrderedDict
import argparse
from net import Net
from data import build_id_idx, check_seq_len, DataIter
import time
import numpy as np
from subprocess import call
import sys

def main():
    options = OrderedDict()

    options['input_dim']          = 44
    options['target_dim']         = 1
    options['unit_type']          = 'LSTM'     # FC/LSTM/GRU
    options['lstm_peephole']      = True
    options['net_width']          = 512
    options['net_depth']          = 12
    options['batch_size']         = 128
    options['window_size']        = 128
    options['step_size']          = 64
    options['init_scale']         = 0.02
    options['init_use_ortho']     = False
    options['batch_norm']         = False
    # options['batch_norm_decay']   = 0.9
    options['residual_gate']      = True
    options['learn_init_states']  = True
    options['learn_id_embedding'] = False
    # options['id_embedding_dim']   = 16
    options['learn_clock_params'] = False
    options['update_type']        = 'nesterov' # sgd/momentum/nesterov
    options['update_mu']          = 0.9        # for momentum/nesterov
    options['force_type']         = 'adadelta' # vanilla/adadelta/rmsprop/adam
    options['force_ms_decay']     = 0.99       # for adadelta/rmsprop
    # options['force_adam_b1']      = 0.9
    # options['force_adam_b2']      = 0.999
    options['frames_per_epoch']   = 8 * 1024 * 1024
    options['lr_init_val']        = 1e-5
    options['lr_lower_bound']     = 1e-7
    options['lr_decay_rate']      = 0.5
    options['max_retry']          = 10
    options['unroll_scan']        = False      # faster training/slower compile

    # options['clock_t_exp_lo']     = 1.         # for learn_clock_params
    # options['clock_t_exp_hi']     = 6.         # for learn_clock_params
    # options['clock_r_on']         = 0.2        # for learn_clock_params
    # options['clock_leak_rate']    = 0.001      # for learn_clock_params
    # options['grad_norm_clip']     = 2.         # comment out to turn off

    if options['unroll_scan']:
        sys.setrecursionlimit(32 * options['window_size']) # 32 is empirical

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir' , type = str, required = True)
    parser.add_argument('--save_to'  , type = str, required = True)
    parser.add_argument('--load_from', type = str)    
    args = parser.parse_args()

    assert 0 == call(str('mkdir -p ' + args.save_to).split())
    
    # store mean/whitening matrices from Reshaper (remove if inapplicable)
    assert 0 == call(str('cp ' + args.data_dir + '/mean.matrix '
                         + args.save_to).split())
    assert 0 == call(str('cp ' + args.data_dir + '/whitening.matrix '
                         + args.save_to).split())


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
    maxlen = max([len(v) for v in options.keys()])
    for k, v in options.iteritems():
        print('    ' + k.ljust(maxlen) + ' : ' + str(v))

    print_hline() # -----------------------------------------------------------
    print('Stats')
    seed = np.random.randint(np.iinfo(np.int32).max)
    np.random.seed(seed)
    print('    Random seed     : ' + str(seed).rjust(10))

    # store sequence length
    print('    Sequence length : ', end = '')
    options['seq_len'] = check_seq_len(args.data_dir + '/train.list',
                                       options['input_dim'],
                                       options['target_dim'])
    assert options['seq_len'] == check_seq_len(args.data_dir + '/dev.list',
                                               options['input_dim'],
                                               options['target_dim'])
    print(str(options['seq_len']).rjust(10))

    def n_seqs(list_file):
        with open(list_file) as f:
            return sum(1 for line in f)

    print('    # of train seqs : '
          + str(n_seqs(args.data_dir + '/train.list')).rjust(10))
    print('    # of dev seqs   : '
          + str(n_seqs(args.data_dir + '/dev.list')).rjust(10))

    # store ID count & internal ID order
    id_idx = build_id_idx(args.data_dir + '/train.list')
    options['id_count'] = len(id_idx)
    with open(args.save_to + '/ids.order', 'w') as f:
        f.write(';'.join(id_idx.iterkeys())) # code_0;...;code_N-1
    print('    # of unique IDs : ' + str(options['id_count']).rjust(10))

    print('    # of weights    : ', end = '')
    net = Net(options, args.save_to, args.load_from)
    print(str(net.n_weights()).rjust(10))

    print_hline() # -----------------------------------------------------------
    print('Compiling fwd/bwd propagators... ', end = '')
    start = time.time()
    f_fwd_bwd_propagate = net.compile_f_fwd_bwd_propagate()
    f_fwd_propagate     = net.compile_f_fwd_propagate()
    print(lapse_from(start))

    print('Compiling updater/initializer... ', end = '')
    start = time.time()
    f_update_v_params = net.compile_f_update_v_params()
    f_initialize_optimizer = net.compile_f_initialize_optimizer()
    print(lapse_from(start))

    # NOTE: window_size must be the same as that given to Net
    train_data = DataIter(list_file   = args.data_dir + '/train.list',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          seq_len     = options['seq_len'],
                          batch_size  = options['batch_size'],
                          input_dim   = options['input_dim'],
                          target_dim  = options['target_dim'],
                          id_idx      = id_idx)
    dev_data   = DataIter(list_file  = args.data_dir + '/dev.list',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          seq_len     = options['seq_len'],
                          batch_size  = options['batch_size'],
                          input_dim   = options['input_dim'],
                          target_dim  = options['target_dim'],
                          id_idx      = id_idx)
    
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

        data_iter.discard_unfinished()
        data_iter.set_step_size(step_size)

        loss_sum = 0.
        frames_seen = 0

        for input_tbi, target_tbi, time_t, id_idx_tb in data_iter:
            if is_training:
                loss = f_fwd_bwd_propagate(input_tbi, target_tbi, 
                                           time_t, id_idx_tb, step_size)
            else:
                loss = f_fwd_propagate(input_tbi, target_tbi, 
                                       time_t, id_idx_tb, step_size)
            
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

    total_trained_frames = 0
    total_trained_frames_at_best = 0
    total_trained_frames_at_pivot = 0
    total_discarded_frames = 0

    loss_pivot = 0.
    loss_prev  = 0.
    loss_best  = 0.

    cur_retry = 0

    net.save_to_workspace(name_prev)
    net.save_to_workspace(name_best)

    lr = options['lr_init_val']
    f_initialize_optimizer()

    while True:
        print_hline() # -------------------------------------------------------
        print('Training...   ', end = '')
        start = time.time()
        loss_train = run_epoch(train_data, lr)
        print(lapse_from(start))

        total_trained_frames += trained_frames_per_epoch

        print('Evaluating... ', end = '')
        start = time.time()
        loss_cur = run_epoch(dev_data, None)
        print(lapse_from(start))

        print('Total trained frames   : '
              + str(total_trained_frames  ).rjust(12))
        print('Total discarded frames : '
              + str(total_discarded_frames).rjust(12))

        print('Train set loss : %.6f' % loss_train)
        print('Dev set loss   : %.6f' % loss_cur, end = '')

        if np.isnan(loss_cur):
            loss_cur = np.float32('inf')
        
        if total_trained_frames == trained_frames_per_epoch or \
               loss_cur < loss_best:
            print(' (best)', end = '')

            loss_best = loss_cur
            total_trained_frames_at_best = total_trained_frames
            net.save_to_workspace(name_best)
        print('')

        if total_trained_frames > trained_frames_per_epoch and \
               loss_prev < loss_cur:
            print_hline() # ---------------------------------------------------
            
            cur_retry += 1
            if cur_retry > options['max_retry']:
                cur_retry = 0

                lr *= options['lr_decay_rate']

                if lr < options['lr_lower_bound']:
                    break
                
                net.load_from_workspace(name_pivot)
                net.save_to_workspace(name_prev)

                discarded_frames \
                    = total_trained_frames - total_trained_frames_at_pivot
                
                print('Discard recently trained '
                      + str(discarded_frames) + ' frames')
                print('New learning rate : ' + str(lr))
                
                f_initialize_optimizer()

                total_discarded_frames += discarded_frames
                total_trained_frames = total_trained_frames_at_pivot
                loss_prev = loss_pivot
            else:
                print('Retry count : ' + str(cur_retry)
                      + ' / ' + str(options['max_retry']))
        else:
            cur_retry = 0

            # prev goes to pivot & cur goes to prev
            loss_pivot, loss_prev = loss_prev, loss_cur
            name_pivot, name_prev = name_prev, name_pivot

            net.save_to_workspace(name_prev)

            total_trained_frames_at_pivot \
                = total_trained_frames - trained_frames_per_epoch
    
    net.load_from_workspace(name_best)
    net.remove_from_workspace(name_pivot)
    net.remove_from_workspace(name_prev)

    total_discarded_frames \
        += total_trained_frames - total_trained_frames_at_best
    total_trained_frames = total_trained_frames_at_best

    print('')
    print('Best network')
    print('Total trained frames   : ' + str(total_trained_frames  ).rjust(12))
    print('Total discarded frames : ' + str(total_discarded_frames).rjust(12))

    loss_train = run_epoch(train_data, None)
    print('Train set loss : %.6f' % loss_train)
    loss_dev = run_epoch(dev_data, None)
    print('Dev set loss   : %.6f' % loss_dev)
    print('')
    
if __name__ == '__main__':
    main()
