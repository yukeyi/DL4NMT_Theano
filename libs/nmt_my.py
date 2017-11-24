u"""
Build a neural machine translation model with soft attention
"""

import cPickle as pkl
import copy
import os
import sys
import time
import math
from pprint import pprint

import numpy as np
import theano
import theano.tensor as tensor

from .constants import profile, fX, NaNReloadPrevious
from .utility.data_iterator import TextIterator
from .utility.optimizers import Optimizers
from .utility.utils import *

from .utility.translate import translate_dev_get_bleu
from .models import NMTModel, TrgAttnNMTModel, TModel
from buffer import Buffer


def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False):
    """Calculate the log probablities on a given corpus using translation model"""

    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        lengths = np.array([len(s) for s in x])

        x, x_mask, y, y_mask = prepare_data(x, y)

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        if normalize:
            pprobs = pprobs / lengths

        for pp in pprobs:
            probs.append(pp)

        sys.stdout.write('\rDid ' + str(n_done) + ' samples')

    print
    return np.array(probs)


def validation(iterator, f_cost, use_noise):
    if not isinstance(iterator, list):
        orig_noise = use_noise.get_value()
        use_noise.set_value(0.)
    
        valid_cost = 0.0
        valid_count = 0
        for x, y in iterator:
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=None)
    
            if x is None:
                continue
    
            valid_cost += f_cost(x, x_mask, y, y_mask) * x_mask.shape[1]
            valid_count += x_mask.shape[1]
    
        use_noise.set_value(orig_noise)
    
        return valid_cost / valid_count
    
    else: # zhen
        sum_valid_cost = 0.0
        sum_valid_count = 0
        for i in range(0, 4):
            valid_cost = 0.0
            valid_count = 0
            for x, y in iterator[i]:
                x, x_mask, y, y_mask = prepare_data(x, y, maxlen=None)            
                if x is None:
                    continue
        
                valid_cost += f_cost(x, x_mask, y, y_mask) * x_mask.shape[1]
                valid_count += x_mask.shape[1]
        
            sum_valid_cost += valid_cost
            sum_valid_count += valid_count
        
        return sum_valid_cost / sum_valid_count

#def combine_feature(cost_value_CE, cost_value_M, cost_value_F, cost_value_B,iteration):
#    feature = [cost_value_CE, cost_value_M, cost_value_F, cost_value_B, iteration]
#    return feature

def abstruct_valid_data(data_buffer,state_batchsize):
    data_number = len(data_buffer[0])*len(data_buffer[0][0])
    ratio = float(state_batchsize) / data_number
    choise = []
    for i in range(0,data_number):
        if(np.random.random() < ratio):
            choise.append(i)
    if(len(choise) > state_batchsize):
        choise = choise[:state_batchsize]
    elif(len(choise) < state_batchsize):
        for i in range(0,data_number):
            if(i not in choise):
                choise.append(i)

    choosed_x = []
    choosed_x_mask = []
    choosed_y = []
    choosed_y_mask = []
    for i in choise:
        group = i / len(data_buffer[0][0])
        order = i % len(data_buffer[0][0])
        choosed_x.append(data_buffer[0][group][order])
        choosed_x_mask.append(data_buffer[1][group][order])
        choosed_y.append(data_buffer[2][group][order])
        choosed_y_mask.append(data_buffer[3][group][order])

    lengths_data_x = [len(s) for s in choosed_x]
    max_length_x = max(lengths_data_x)

    lengths_data_y = [len(s) for s in choosed_y]
    max_length_y = max(lengths_data_y)

    x = np.zeros((state_batchsize, max_length_x)).astype('int64')
    y = np.zeros((state_batchsize, max_length_y)).astype('int64')
    x_mask = np.zeros((state_batchsize, max_length_x)).astype('float32')
    y_mask = np.zeros((state_batchsize, max_length_y)).astype('float32')

    for i in range(0,state_batchsize):
        x[i][:len(choosed_x[i])] = choosed_x[i]
        x_mask[i][:len(choosed_x_mask[i])] = choosed_x_mask[i]
        y[i][:len(choosed_y[i])] = choosed_y[i]
        y_mask[i][:len(choosed_y_mask[i])] = choosed_y_mask[i]


    return np.array(x).T, \
           np.array(x_mask).T, \
           np.array(y).T, \
           np.array(y_mask).T

def abstruct_partial_data(x, y, ratio):
    length = len(x)
    choosed_x = []
    choosed_y = []
    for i in range(0,length):
        if(np.random.random() < ratio):
            choosed_x.append(x[i])
            choosed_y.append(y[i])
    return choosed_x, choosed_y

def train(teacher = None,
          update_frequence_batches = 10,
          state_batchsize = 32,
          dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          n_words_src=30000,
          n_words=30000,
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=1.,  # learning rate
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=80,
          saveto='model.npz',
          saveFreq=1000,  # save the parameters after every saveFreq updates
          validFreq=2500,
          dev_bleu_freq=20000,
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          valid_datasets=('./data/dev/dev_en.tok',
                          './data/dev/dev_fr.tok'),
          small_train_datasets=('./data/train/small_en-fr.en',
                                './data/train/small_en-fr.fr'),
          use_dropout=False,
          dropout_out = False,
          reload_=False,
          overwrite=False,
          preload='',

          # Options below are from v-yanfa
          dump_before_train=True,
          plot_graph=None,
          vocab_filenames=('./data/dic/filtered_dic_en-fr.en.pkl',
                           './data/dic/filtered_dic_en-fr.fr.pkl'),
          map_filename='./data/dic/mapFullVocab2Top1MVocab.pkl',
          lr_discount_freq=80000,

          # Options of deeper encoder and decoder
          n_encoder_layers=1,
          n_decoder_layers=1,
          encoder_many_bidirectional=True,

          attention_layer_id=0,
          unit='gru',
          residual_enc=None,
          residual_dec=None,
          use_zigzag=False,

          initializer='orthogonal',
          given_embedding=None,

          dist_type=None,
          dist_recover_lr_iter=False,

          unit_size=2,
          cond_unit_size=2,

          given_imm=False,
          dump_imm=False,
          shuffle_data=False,

          decoder_all_attention=False,
          average_context=False,
          task='en-fr',

          fine_tune_patience=8,
          fine_tune_type = 'cost',
          nccl = False,
          src_vocab_map_file = None,
          tgt_vocab_map_file = None,

          trg_attention_layer_id=None,
          fix_dp_bug = False,
          io_buffer_size = 40,
          start_epoch = 0,
          start_from_histo_data = False,
          zhen = False,
          previous_best_bleu = 0.0,
          previous_best_valid_cost = 1e5,
          previous_bad_count = 0,
          previous_finetune_cnt = 0,
          ):
    model_options = locals().copy()

    # Set distributed computing environment
    worker_id = 0
    if dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            from . import multiverso_ as mv

        worker_id = mv.worker_id()
    elif dist_type == 'mpi_reduce':
        from mpi4py import MPI
        mpi_communicator = MPI.COMM_WORLD
        worker_id = mpi_communicator.Get_rank()
        workers_cnt = mpi_communicator.Get_size()

        if nccl:
            nccl_comm = init_nccl_env(mpi_communicator)

    print 'Use {}, worker id: {}'.format('multiverso' if dist_type == 'mv' else 'mpi' if dist_recover_lr_iter else 'none', worker_id)
    sys.stdout.flush()

    # Set logging file
    set_logging_file('log/complete/e{}d{}_res{}_att{}_worker{}_task{}_{}.txt'.format(
        n_encoder_layers, n_decoder_layers, residual_enc, attention_layer_id,
        worker_id, task, time.strftime('%m-%d-%H-%M-%S'),
    ))

    log('''\
Start Time = {}
'''.format(
        time.strftime('%c'),
    ))

    # Model options: load and save
    if worker_id == 0:
        message('Top options:')
        pprint(model_options)
        pprint(model_options, stream=get_logging_file())
        message('Done')
    sys.stdout.flush()

    load_options_train(model_options, reload_, preload)
    check_options(model_options)
    model_options['cost_normalization'] = 1
    ada_alpha = 0.95
    if dist_type == 'mpi_reduce':
        model_options['cost_normalization'] = workers_cnt

    if worker_id == 0:
        message('Model options:')
        pprint(model_options)
        pprint(model_options, stream=get_logging_file())
        message()

    print 'Loading data'
    log('\n\n\nStart to prepare data\n@Current Time = {}'.format(time.time()))
    sys.stdout.flush()

    dataset_src, dataset_tgt = datasets[0], datasets[1]
    # randomly select partial data
    #dataset_src, dataset_tgt = abstruct_partial_data(dataset_src, dataset_tgt, 0.3)


    if shuffle_data:
        text_iterator_list = [None for _ in range(10)]
        text_iterator = None
    else:
        text_iterator_list = None
        text_iterator = TextIterator(
            dataset_src, dataset_tgt,
            vocab_filenames[0], vocab_filenames[1],
            batch_size,n_words_src, n_words,maxlen, k = io_buffer_size,
        )

    if not zhen:
        valid_iterator=TextIterator(
            valid_datasets[0], valid_datasets[1],
            vocab_filenames[0], vocab_filenames[1],
            valid_batch_size, n_words_src, n_words, k = io_buffer_size,
        )
    else:
        valid_iterator = []
        for i in range(4, 8): # NIST2005.reference4-7
            valid_iterator.append(TextIterator(
                valid_datasets[0], valid_datasets[1] if not zhen else valid_datasets[2]+'{}'.format(i),
                vocab_filenames[0], vocab_filenames[1],
                valid_batch_size, n_words_src, n_words,k = io_buffer_size,
            ))

    small_train_iterator = TextIterator(
        small_train_datasets[0], small_train_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        valid_batch_size, n_words_src, n_words, k = io_buffer_size,
    )

    print 'Building model'
    if trg_attention_layer_id is None:
        model = NMTModel(model_options)
    else:
        model = TrgAttnNMTModel(model_options)

    params = model.initializer.init_params()

    # Reload parameters
    if reload_ and os.path.exists(preload):
        print 'Reloading model parameters'
        load_params(preload, params, src_map_file = src_vocab_map_file, tgt_map_file = tgt_vocab_map_file)
    sys.stdout.flush()

    # Given embedding
    if given_embedding is not None:
        print 'Loading given embedding...',
        load_embedding(params, given_embedding)
        print 'Done'

    if worker_id == 0:
        print_params(params)
    model.init_tparams(params)

    # Build model
    trng, use_noise, \
    x, x_mask, y, y_mask, y_hat_reward, predicted_y, predicted_y_mask, predicted_probs,\
    opt_ret, \
    cost_CE_flatten, test_cost_CE, \
    cost_F_flatten, test_cost_F, \
    cost_M_flatten, test_cost_M, \
    cost_B_flatten, test_cost_B, \
    x_emb, probs, alpha = model.build_model()

    inps = [x, x_mask, y, y_mask]
    inps_B = [x, x_mask, y, y_mask, y_hat_reward]
    inps_M = [x, x_mask, y, y_mask, predicted_y, predicted_y_mask, predicted_probs, alpha]

    print 'Building sampler'
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs_CE = theano.function(inps, probs, profile=profile)
    f_log_probs_F = theano.function(inps, probs, profile=profile)
    f_log_probs_M = theano.function(inps, probs, profile=profile)
    f_log_probs_B = theano.function(inps_B, probs, profile=profile, on_unused_input='warn')
    print 'Done'
    sys.stdout.flush()

    test_cost_CE = test_cost_CE.mean() #FIXME: do not regularize test_cost here
    cost_CE = cost_CE_flatten.mean()
    cost_CE = l2_regularization(cost_CE, model.P, decay_c)
    cost_CE = regularize_alpha_weights(cost_CE, alpha_c, model_options, x_mask, y_mask, opt_ret)
    test_cost_F = test_cost_F.mean() #FIXME: do not regularize test_cost here
    cost_F = cost_F_flatten.mean()
    cost_F = l2_regularization(cost_F, model.P, decay_c)
    cost_F = regularize_alpha_weights(cost_F, alpha_c, model_options, x_mask, y_mask, opt_ret)
    test_cost_M = test_cost_M.mean() #FIXME: do not regularize test_cost here
    cost_M = cost_M_flatten.mean()
    cost_M = l2_regularization(cost_M, model.P, decay_c)
    cost_M = regularize_alpha_weights(cost_M, alpha_c, model_options, x_mask, y_mask, opt_ret)
    test_cost_B = test_cost_B.mean() #FIXME: do not regularize test_cost here
    cost_B = cost_B_flatten.mean()
    cost_B = l2_regularization(cost_B, model.P, decay_c)
    cost_B = regularize_alpha_weights(cost_B, alpha_c, model_options, x_mask, y_mask, opt_ret)

    print 'Building f_cost...',
    f_cost_CE = theano.function(inps, test_cost_CE, profile=profile)
    f_cost_F = theano.function(inps, test_cost_F, profile=profile)
    f_cost_M = theano.function(inps_M, test_cost_M, profile=profile)
    f_cost_B = theano.function(inps_B, test_cost_B, profile=profile)
    #f_cost_flatten_CE = theano.function(inps, cost_CE_flatten, profile=profile)
    #f_cost_flatten_F = theano.function(inps, cost_F_flatten, profile=profile)
    #f_cost_flatten_M = theano.function(inps_M, cost_M_flatten, profile=profile)
    #f_cost_flatten_B = theano.function(inps_B, cost_B_flatten, profile=profile)
    print 'Done'

    if plot_graph is not None:
        print 'Plotting post-compile graph...',
        theano.printing.pydotprint(
            f_cost_CE,
            outfile='pictures/CE/post_compile_{}'.format(plot_graph),
            var_with_name_simple=True,
        )
        theano.printing.pydotprint(
            f_cost_M,
            outfile='pictures/M/post_compile_{}'.format(plot_graph),
            var_with_name_simple=True,
        )
        theano.printing.pydotprint(
            f_cost_F,
            outfile='pictures/F/post_compile_{}'.format(plot_graph),
            var_with_name_simple=True,
        )
        theano.printing.pydotprint(
            f_cost_B,
            outfile='pictures/B/post_compile_{}'.format(plot_graph),
            var_with_name_simple=True,
        )
        print 'Done'

    print 'Computing gradient...',
    grads_CE = tensor.grad(cost_CE, wrt=itemlist(model.P))
    grads_M = tensor.grad(cost_M, wrt=itemlist(model.P))
    grads_F = tensor.grad(cost_F, wrt=itemlist(model.P))
    grads_B = tensor.grad(cost_B, wrt=itemlist(model.P))

    clip_shared = theano.shared(np.array(clip_c, dtype=fX), name='clip_shared')

    if dist_type != 'mpi_reduce': #build grads clip into computational graph
        grads_CE, g2_CE = clip_grad_remove_nan(grads_CE, clip_shared, model.P)
        grads_M, g2_M = clip_grad_remove_nan(grads_M, clip_shared, model.P)
        grads_F, g2_F = clip_grad_remove_nan(grads_F, clip_shared, model.P)
        grads_B, g2_B = clip_grad_remove_nan(grads_B, clip_shared, model.P)
    else: #do the grads clip after gradients aggregation
        g2_CE = None
        g2_M = None
        g2_F = None
        g2_B = None

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',

    uidx = search_start_uidx(reload_, preload)
    given_imm_data = get_optimizer_imm_data(optimizer, given_imm, preload, uidx)

    f_grad_shared_CE, f_update_CE, grads_shared_CE, imm_shared_CE = Optimizers[optimizer](
        lr, model.P, grads_CE, inps, cost_CE, g2=g2_CE, given_imm_data=given_imm_data, alpha = ada_alpha)
    f_grad_shared_M, f_update_M, grads_shared_M, imm_shared_M = Optimizers[optimizer](
        lr, model.P, grads_M, inps_M, cost_M, g2=g2_M, given_imm_data=given_imm_data, alpha = ada_alpha)
    f_grad_shared_F, f_update_F, grads_shared_F, imm_shared_F = Optimizers[optimizer](
        lr, model.P, grads_F, inps, cost_F, g2=g2_F, given_imm_data=given_imm_data, alpha = ada_alpha)
    f_grad_shared_B, f_update_B, grads_shared_B, imm_shared_B = Optimizers[optimizer](
        lr, model.P, grads_B, inps_B, cost_B, g2=g2_B, given_imm_data=given_imm_data, alpha = ada_alpha)
    print 'Done'

    if dist_type == 'mpi_reduce':
        f_grads_clip_CE = make_grads_clip_func(grads_shared = grads_shared_CE, mt_tparams= model.P, clip_c_shared = clip_shared)
        f_grads_clip_M = make_grads_clip_func(grads_shared = grads_shared_M, mt_tparams= model.P, clip_c_shared = clip_shared)
        f_grads_clip_F = make_grads_clip_func(grads_shared = grads_shared_F, mt_tparams= model.P, clip_c_shared = clip_shared)
        f_grads_clip_B = make_grads_clip_func(grads_shared = grads_shared_B, mt_tparams= model.P, clip_c_shared = clip_shared)

    print 'Optimization'
    log('Preparation Done\n@Current Time = {}'.format(time.time()))

    if dist_type == 'mv':
        mv.barrier()
    elif dist_type == 'mpi_reduce':
        #create receive buffers for mpi allreduce
        rec_grads_CE = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]
        rec_grads_M = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]
        rec_grads_F = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]
        rec_grads_B = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]

    estop = False
    history_errs = []
    best_bleu = -1.0
    best_valid_cost = 1e6
    best_p = None
    bad_counter = 0

    epoch_n_batches = 0
    pass_batches = 0

    start_uidx = uidx

    if dump_before_train:
        print 'Dumping before train...',
        saveto_uidx = '{}.iter{}.npz'.format(
            os.path.splitext(saveto)[0], uidx)
        np.savez(saveto_uidx, history_errs=history_errs,
                 uidx=uidx, **unzip(model.P))
        save_options(model_options, uidx, saveto)
        print 'Done'
        sys.stdout.flush()

    #sync all model parameters if train from scratch
    if not reload_ and dist_type == 'mpi_reduce':
        all_reduce_params_nccl(nccl_comm, itemlist(model.P))
        for t_value in itemlist(model.P):
            t_value.set_value(t_value.get_value() / workers_cnt)

    best_valid_cost = validation(valid_iterator, f_cost_CE, use_noise)
    small_train_cost = validation(small_train_iterator, f_cost_CE, use_noise)
    best_bleu = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise, zhen = zhen) if reload_ else 0
    message('Worker id {}, Initial Valid cost {:.5f} Small train cost {:.5f} Valid BLEU {:.2f}'.format(worker_id, best_valid_cost, small_train_cost, best_bleu))

    best_bleu = previous_best_bleu
    best_valid_cost = previous_best_valid_cost #do not let initial state affect the training process
    bad_counter = previous_bad_count

    commu_time_sum = 0.0
    cp_time_sum = 0.0
    reduce_time_sum = 0.0

    start_time = time.time()
    finetune_cnt = previous_finetune_cnt
    last_saveto_paths = []

    if start_from_histo_data:
        if uidx != 0:
            epoch_n_batches = get_epoch_batch_cnt(dataset_src, dataset_tgt, vocab_filenames, batch_size, maxlen, n_words_src, n_words) \
                if worker_id == 0 else None
        else:
            epoch_n_batches = 1 #avoid heavy data IO

        if dist_type == 'mpi_reduce':
            epoch_n_batches = mpi_communicator.bcast(epoch_n_batches, root = 0)

        start_epoch = start_epoch + uidx / epoch_n_batches
        pass_batches = uidx % epoch_n_batches

    print 'worker', worker_id, 'uidx', uidx, 'l_rate', lrate, 'ada_alpha', ada_alpha, 'n_batches', epoch_n_batches, \
        'start_epoch', start_epoch, 'pass_batches', pass_batches

    print 'Allocating GPU memory in advance for batch data...',
    x, x_mask, y, y_mask = get_batch_place_holder(batch_size, maxlen)
    if dist_type != 'mpi_reduce':
        cost_CE, g2_value_CE = f_grad_shared_CE(x, x_mask, y, y_mask)
    else:
        cost_CE = f_grad_shared_CE(x, x_mask, y, y_mask)
    f_update_CE(np.float32(.0))

    # initial used loss function
    loss_name = 'F'
    r_value = 0.9
    # data buffer
    data_buffer = [[],[],[],[]]
    # return value
    value_return = {}
    value_return['state'] = []
    value_return['action'] = []


    for eidx in xrange(start_epoch, max_epochs):

        if shuffle_data:
            text_iterator = load_shuffle_text_iterator(
                eidx, worker_id, text_iterator_list,
                datasets, vocab_filenames, batch_size, maxlen, n_words_src, n_words, buffer_size=io_buffer_size
            )
        n_samples = 0
        if dist_type == 'mpi_reduce':
            mpi_communicator.Barrier()

        for i, (x, y) in enumerate(text_iterator):
            if eidx == start_epoch and i < pass_batches: #ignore the first several batches when reload
                continue
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            if(loss_name == 'B'):
                y_hats, y_hat_rewards = model.get_rl_reward(x, y)
                x, x_mask, y, y_mask, y_costs = prepare_data(x, y_hats, maxlen=sys.maxint, seqs_y_hat_scores= y_hat_rewards)
            elif(loss_name == 'M'):
                predicted_y, _ = model.get_rl_reward(x, y)
                original_x = copy.deepcopy(x)
                x, x_mask, y, y_mask = prepare_data(original_x, y, maxlen=maxlen)
                _, _, predicted_y, predicted_y_mask = prepare_data(original_x, predicted_y, maxlen=sys.maxint)
                predicted_probs = f_log_probs_M(x, x_mask, predicted_y, predicted_y_mask)
            else:
                x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            data_buffer[0].append(x.T)
            data_buffer[1].append(x_mask.T)
            data_buffer[2].append(y.T)
            data_buffer[3].append(y_mask.T)

            effective_uidx = uidx - start_uidx
            ud_start = time.time()


            if(uidx % update_frequence_batches == 0):
                # abstruct data for choosing loss function
                valid_x, valid_x_mask, valid_y, valid_y_mask = abstruct_valid_data(data_buffer,state_batchsize)
                data_buffer = [[],[],[],[]]

                # abstruct state for teacher model
                cost_value_CE = f_cost_CE(valid_x, valid_x_mask, valid_y, valid_y_mask) / valid_y_mask.shape[0] # scalar

                predicted_y, _ = model.get_rl_reward(valid_x.T, valid_y.T)
                _, _, predicted_y, predicted_y_mask = prepare_data(valid_x.T, predicted_y, maxlen=sys.maxint)
                predicted_probs = f_log_probs_M(valid_x, valid_x_mask, predicted_y, predicted_y_mask)
                cost_value_M01 = f_cost_M(valid_x, valid_x_mask, valid_y, valid_y_mask, predicted_y, predicted_y_mask,
                                          predicted_probs, 0.1) / valid_y_mask.shape[0]
                cost_value_M03 = f_cost_M(valid_x, valid_x_mask, valid_y, valid_y_mask, predicted_y, predicted_y_mask,
                                          predicted_probs, 0.3) / valid_y_mask.shape[0]
                cost_value_M05 = f_cost_M(valid_x, valid_x_mask, valid_y, valid_y_mask, predicted_y, predicted_y_mask,
                                          predicted_probs, 0.5) / valid_y_mask.shape[0]
                cost_value_M07 = f_cost_M(valid_x, valid_x_mask, valid_y, valid_y_mask, predicted_y, predicted_y_mask,
                                          predicted_probs, 0.7) / valid_y_mask.shape[0]
                cost_value_M09 = f_cost_M(valid_x, valid_x_mask, valid_y, valid_y_mask, predicted_y, predicted_y_mask,
                                          predicted_probs, 0.9) / valid_y_mask.shape[0]

                cost_value_F = f_cost_F(valid_x, valid_x_mask, valid_y, valid_y_mask) / valid_y_mask.shape[0]

                y_hats, y_hat_rewards = model.get_rl_reward(valid_x.T, valid_y.T)
                _, _, y, y_mask, y_costs = prepare_data(valid_x.T, y_hats, maxlen=sys.maxint, seqs_y_hat_scores= y_hat_rewards)
                cost_value_B = f_cost_B(valid_x, valid_x_mask, y, y_mask, y_costs) / y_mask.shape[0]

                iteration = float(eidx)        # iteration feature

                # combine features
                teacher_state = [cost_value_CE, cost_value_M01, cost_value_M03, cost_value_M05, cost_value_M07, cost_value_M09,
                                 cost_value_F, cost_value_B, iteration]

                # predict name of loss function chosen for next big batch
                prob = teacher.predict(np.array([teacher_state]).astype('float32'))[0][0].tolist()
                maxprob_index = prob.index(max(prob))
                value_return['state'].append(teacher_state)
                value_return['action'].append(maxprob_index)

                print(maxprob_index)
                maxprob_index = 0

                if maxprob_index == 0:
                    loss_name = 'CE'
                elif maxprob_index == 1:
                    loss_name = 'B'
                elif maxprob_index == 2:
                    loss_name = 'F'
                else:
                    loss_name = 'M'
                    if(maxprob_index == 3):
                        r_value = 0.1
                    if(maxprob_index == 4):
                        r_value = 0.3
                    if(maxprob_index == 5):
                        r_value = 0.5
                    if(maxprob_index == 6):
                        r_value = 0.7
                    if(maxprob_index == 7):
                        r_value = 0.9


            if(loss_name == 'CE'):
                # compute cost, grads
                if dist_type != 'mpi_reduce':
                    cost_CE, g2_value_CE = f_grad_shared_CE(x, x_mask, y, y_mask)
                else:
                    cost_CE = f_grad_shared_CE(x, x_mask, y, y_mask)

                if dist_type == 'mpi_reduce':
                    reduce_start = time.time()
                    commu_time = 0
                    gpucpu_cp_time = 0
                    if not nccl:
                        commu_time, gpucpu_cp_time = all_reduce_params(grads_shared_CE, rec_grads_CE)
                    else:
                        commu_time, gpucpu_cp_time = all_reduce_params_nccl(nccl_comm, grads_shared_CE)
                    reduce_time = time.time() - reduce_start
                    commu_time_sum += commu_time
                    reduce_time_sum += reduce_time
                    cp_time_sum += gpucpu_cp_time

                    g2_value_CE = f_grads_clip_CE()

                curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx \
                    else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
                if curr_lr < lrate:
                    print 'Curr lr {:.3f}'.format(curr_lr)

                if np.isnan(cost_CE) or np.isinf(cost_CE):
                    message('NaN detected')
                    sys.stdout.flush()
                    clip_shared.set_value(np.float32(clip_shared.get_value() * 0.8))
                    message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))

                    # reload the N-th previous saved model.
                    reload_iter = (uidx // saveFreq - NaNReloadPrevious + 1) * saveFreq

                    if reload_iter < saveFreq:
                        # if not exist, reload the first saved model.
                        reload_iter = saveFreq

                    can_reload = False
                    while reload_iter < uidx:
                        model_save_path = '{}.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)
                        imm_save_path = '{}_imm.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)

                        can_reload = True
                        if not os.path.exists(model_save_path):
                            message('No saved model at {}'.format(model_save_path))
                            can_reload = False
                        if not os.path.exists(imm_save_path):
                            message('No saved immediate file at {}'.format(imm_save_path))
                            can_reload = False

                        if can_reload:
                            # find the model to reload.
                            message('Load previously dumped model at {}, immediate at {}'.format(
                                model_save_path, imm_save_path))
                            prev_params = load_params(model_save_path, params)
                            zipp(prev_params, model.P)
                            prev_imm_data = get_optimizer_imm_data(optimizer, True, saveto, reload_iter)
                            set_optimizer_imm_data(optimizer, prev_imm_data, imm_shared_CE)

                            #begin scale the model parameters
                            for (p, grad) in zip(itemlist(model.P), grads_shared_CE):
                                grad.set_value(p.get_value() * np.float32(.1))
                            break

                        reload_iter += saveFreq

                    if not can_reload:
                        message('Cannot reload any saved model. Task exited')
                        return 1., 1., 1.

                # do the update on parameters
                f_update_CE(curr_lr)

                ud = time.time() - ud_start

                # discount learning rate
                # FIXME: Do NOT enable this and fine-tune at the same time
                if lr_discount_freq > 0 and np.mod(effective_uidx, lr_discount_freq) == 0:
                    lrate *= 0.5
                    message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

                # sync batch
                if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                    comm_start = time.time()
                    model.sync_tparams()
                    message('@Comm time = {:.5f}'.format(time.time() - comm_start))

                # verbose
                if np.mod(uidx, dispFreq) == 0:
                    message('Worker {} Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                        worker_id, eidx, uidx, float(cost_CE), float(g2_value_CE), ud, time.time() - start_time,
                    ))
                    sys.stdout.flush()

            elif(loss_name == 'M'):
                # compute cost, grads
                if dist_type != 'mpi_reduce':
                    cost_M, g2_value_M = f_grad_shared_M(x, x_mask, y, y_mask, predicted_y, predicted_y_mask, predicted_probs, r_value)
                else:
                    cost_M = f_grad_shared_M(x, x_mask, y, y_mask, predicted_y, predicted_y_mask, predicted_probs, r_value)

                if dist_type == 'mpi_reduce':
                    reduce_start = time.time()
                    commu_time = 0
                    gpucpu_cp_time = 0
                    if not nccl:
                        commu_time, gpucpu_cp_time = all_reduce_params(grads_shared_M, rec_grads_M)
                    else:
                        commu_time, gpucpu_cp_time = all_reduce_params_nccl(nccl_comm, grads_shared_M)
                    reduce_time = time.time() - reduce_start
                    commu_time_sum += commu_time
                    reduce_time_sum += reduce_time
                    cp_time_sum += gpucpu_cp_time

                    g2_value_M = f_grads_clip_M()

                curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx \
                    else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
                if curr_lr < lrate:
                    print 'Curr lr {:.3f}'.format(curr_lr)

                if np.isnan(cost_M) or np.isinf(cost_M):
                    message('NaN detected')
                    sys.stdout.flush()
                    clip_shared.set_value(np.float32(clip_shared.get_value() * 0.8))
                    message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))

                    # reload the N-th previous saved model.
                    reload_iter = (uidx // saveFreq - NaNReloadPrevious + 1) * saveFreq

                    if reload_iter < saveFreq:
                        # if not exist, reload the first saved model.
                        reload_iter = saveFreq

                    can_reload = False
                    while reload_iter < uidx:
                        model_save_path = '{}.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)
                        imm_save_path = '{}_imm.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)

                        can_reload = True
                        if not os.path.exists(model_save_path):
                            message('No saved model at {}'.format(model_save_path))
                            can_reload = False
                        if not os.path.exists(imm_save_path):
                            message('No saved immediate file at {}'.format(imm_save_path))
                            can_reload = False

                        if can_reload:
                            # find the model to reload.
                            message('Load previously dumped model at {}, immediate at {}'.format(
                                model_save_path, imm_save_path))
                            prev_params = load_params(model_save_path, params)
                            zipp(prev_params, model.P)
                            prev_imm_data = get_optimizer_imm_data(optimizer, True, saveto, reload_iter)
                            set_optimizer_imm_data(optimizer, prev_imm_data, imm_shared_M)

                            # begin scale the model parameters
                            for (p, grad) in zip(itemlist(model.P), grads_shared_M):
                                grad.set_value(p.get_value() * np.float32(.1))
                            break

                        reload_iter += saveFreq

                    if not can_reload:
                        message('Cannot reload any saved model. Task exited')
                        return 1., 1., 1.

                # do the update on parameters
                f_update_M(curr_lr)

                ud = time.time() - ud_start

                # discount learning rate
                # FIXME: Do NOT enable this and fine-tune at the same time
                if lr_discount_freq > 0 and np.mod(effective_uidx, lr_discount_freq) == 0:
                    lrate *= 0.5
                    message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

                # sync batch
                if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                    comm_start = time.time()
                    model.sync_tparams()
                    message('@Comm time = {:.5f}'.format(time.time() - comm_start))

                # verbose
                if np.mod(uidx, dispFreq) == 0:
                    message('Worker {} Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                        worker_id, eidx, uidx, float(cost_M), float(g2_value_M), ud, time.time() - start_time,
                    ))
                    sys.stdout.flush()

            elif(loss_name == 'F'):
                # compute cost, grads
                if dist_type != 'mpi_reduce':
                    cost_F, g2_value_F = f_grad_shared_F(x, x_mask, y, y_mask)
                else:
                    cost_F = f_grad_shared_F(x, x_mask, y, y_mask)

                if dist_type == 'mpi_reduce':
                    reduce_start = time.time()
                    commu_time = 0
                    gpucpu_cp_time = 0
                    if not nccl:
                        commu_time, gpucpu_cp_time = all_reduce_params(grads_shared_F, rec_grads_F)
                    else:
                        commu_time, gpucpu_cp_time = all_reduce_params_nccl(nccl_comm, grads_shared_F)
                    reduce_time = time.time() - reduce_start
                    commu_time_sum += commu_time
                    reduce_time_sum += reduce_time
                    cp_time_sum += gpucpu_cp_time

                    g2_value_F = f_grads_clip_F()

                curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx \
                    else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
                if curr_lr < lrate:
                    print 'Curr lr {:.3f}'.format(curr_lr)

                if np.isnan(cost_F) or np.isinf(cost_F):
                    message('NaN detected')
                    sys.stdout.flush()
                    clip_shared.set_value(np.float32(clip_shared.get_value() * 0.8))
                    message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))

                    # reload the N-th previous saved model.
                    reload_iter = (uidx // saveFreq - NaNReloadPrevious + 1) * saveFreq

                    if reload_iter < saveFreq:
                        # if not exist, reload the first saved model.
                        reload_iter = saveFreq

                    can_reload = False
                    while reload_iter < uidx:
                        model_save_path = '{}.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)
                        imm_save_path = '{}_imm.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)

                        can_reload = True
                        if not os.path.exists(model_save_path):
                            message('No saved model at {}'.format(model_save_path))
                            can_reload = False
                        if not os.path.exists(imm_save_path):
                            message('No saved immediate file at {}'.format(imm_save_path))
                            can_reload = False

                        if can_reload:
                            # find the model to reload.
                            message('Load previously dumped model at {}, immediate at {}'.format(
                                model_save_path, imm_save_path))
                            prev_params = load_params(model_save_path, params)
                            zipp(prev_params, model.P)
                            prev_imm_data = get_optimizer_imm_data(optimizer, True, saveto, reload_iter)
                            set_optimizer_imm_data(optimizer, prev_imm_data, imm_shared_F)

                            # begin scale the model parameters
                            for (p, grad) in zip(itemlist(model.P), grads_shared_F):
                                grad.set_value(p.get_value() * np.float32(.1))
                            break

                        reload_iter += saveFreq

                    if not can_reload:
                        message('Cannot reload any saved model. Task exited')
                        return 1., 1., 1.

                # do the update on parameters
                f_update_F(curr_lr)

                ud = time.time() - ud_start

                # discount learning rate
                # FIXME: Do NOT enable this and fine-tune at the same time
                if lr_discount_freq > 0 and np.mod(effective_uidx, lr_discount_freq) == 0:
                    lrate *= 0.5
                    message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

                # sync batch
                if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                    comm_start = time.time()
                    model.sync_tparams()
                    message('@Comm time = {:.5f}'.format(time.time() - comm_start))

                # verbose
                if np.mod(uidx, dispFreq) == 0:
                    message('Worker {} Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                        worker_id, eidx, uidx, float(cost_F), float(g2_value_F), ud, time.time() - start_time,
                    ))
                    sys.stdout.flush()

            else:
                # compute cost, grads
                if dist_type != 'mpi_reduce':
                    cost_B, g2_value_B = f_grad_shared_B(x, x_mask, y, y_mask, y_costs)
                else:
                    cost_B = f_grad_shared_B(x, x_mask, y, y_mask, y_costs)

                if dist_type == 'mpi_reduce':
                    reduce_start = time.time()
                    commu_time = 0
                    gpucpu_cp_time = 0
                    if not nccl:
                        commu_time, gpucpu_cp_time = all_reduce_params(grads_shared_B, rec_grads_B)
                    else:
                        commu_time, gpucpu_cp_time = all_reduce_params_nccl(nccl_comm, grads_shared_B)
                    reduce_time = time.time() - reduce_start
                    commu_time_sum += commu_time
                    reduce_time_sum += reduce_time
                    cp_time_sum += gpucpu_cp_time

                    g2_value_B = f_grads_clip_B()

                curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx \
                    else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
                if curr_lr < lrate:
                    print 'Curr lr {:.3f}'.format(curr_lr)

                if np.isnan(cost_B) or np.isinf(cost_B):
                    message('NaN detected')
                    sys.stdout.flush()
                    clip_shared.set_value(np.float32(clip_shared.get_value() * 0.8))
                    message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))

                    # reload the N-th previous saved model.
                    reload_iter = (uidx // saveFreq - NaNReloadPrevious + 1) * saveFreq

                    if reload_iter < saveFreq:
                        # if not exist, reload the first saved model.
                        reload_iter = saveFreq

                    can_reload = False
                    while reload_iter < uidx:
                        model_save_path = '{}.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)
                        imm_save_path = '{}_imm.iter{}.npz'.format(os.path.splitext(saveto)[0], reload_iter)

                        can_reload = True
                        if not os.path.exists(model_save_path):
                            message('No saved model at {}'.format(model_save_path))
                            can_reload = False
                        if not os.path.exists(imm_save_path):
                            message('No saved immediate file at {}'.format(imm_save_path))
                            can_reload = False

                        if can_reload:
                            # find the model to reload.
                            message('Load previously dumped model at {}, immediate at {}'.format(
                                model_save_path, imm_save_path))
                            prev_params = load_params(model_save_path, params)
                            zipp(prev_params, model.P)
                            prev_imm_data = get_optimizer_imm_data(optimizer, True, saveto, reload_iter)
                            set_optimizer_imm_data(optimizer, prev_imm_data, imm_shared_B)

                            # begin scale the model parameters
                            for (p, grad) in zip(itemlist(model.P), grads_shared_B):
                                grad.set_value(p.get_value() * np.float32(.1))
                            break

                        reload_iter += saveFreq

                    if not can_reload:
                        message('Cannot reload any saved model. Task exited')
                        return 1., 1., 1.

                # do the update on parameters
                f_update_B(curr_lr)

                ud = time.time() - ud_start

                # discount learning rate
                # FIXME: Do NOT enable this and fine-tune at the same time
                if lr_discount_freq > 0 and np.mod(effective_uidx, lr_discount_freq) == 0:
                    lrate *= 0.5
                    message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

                # sync batch
                if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                    comm_start = time.time()
                    model.sync_tparams()
                    message('@Comm time = {:.5f}'.format(time.time() - comm_start))

                # verbose
                if np.mod(uidx, dispFreq) == 0:
                    message('Worker {} Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                        worker_id, eidx, uidx, float(cost_B), float(g2_value_B), ud, time.time() - start_time,
                    ))
                    sys.stdout.flush()


            if np.mod(uidx, saveFreq) == 0 and worker_id == 0:
                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    model.save_model(saveto, history_errs, uidx)
                    print 'Done'
                    sys.stdout.flush()

                # save immediate data in adadelta
                dump_optimizer_imm_data(optimizer, imm_shared_CE, dump_imm, saveto, uidx)

            if np.mod(uidx, validFreq) == 0:
                valid_cost = validation(valid_iterator, f_cost_CE, use_noise)
                small_train_cost = validation(small_train_iterator, f_cost_CE, use_noise)
                valid_bleu = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise)
                message('Worker {} Valid cost {:.5f} Small train cost {:.5f} Valid BLEU {:.2f} Bad count {}'.format(
                    worker_id, valid_cost, small_train_cost, valid_bleu, bad_counter))
                sys.stdout.flush()

                # Fine-tune based on dev cost or bleu
                if fine_tune_patience > 0:
                    better_perf = False
                    if valid_bleu > best_bleu:
                        if fine_tune_type != 'cost':
                            bad_counter = 0
                            better_perf = True
                        best_bleu = valid_bleu
                    if valid_cost < best_valid_cost:
                        if fine_tune_type == 'cost':
                            bad_counter = 0
                            better_perf = True
                        best_valid_cost = valid_cost

                    if better_perf:
                        # safe sync before dump to make sure the models are the same on every worker
                        if dist_type == 'mpi_reduce':
                            all_reduce_params_nccl(nccl_comm, itemlist(model.P))
                            for t_value in itemlist(model.P):
                                t_value.set_value(t_value.get_value() / workers_cnt)
                        # dump the best model so far, including the immediate file
                        if worker_id == 0:
                            message('Dump the the best model so far at uidx {}'.format(uidx))
                            model.save_model(saveto, history_errs)
                            dump_optimizer_imm_data(optimizer, imm_shared_CE, dump_imm, saveto)
                    else:
                        bad_counter += 1
                        if bad_counter >= fine_tune_patience:
                            print 'Fine tune:',
                            if finetune_cnt % 2 == 0:
                                lrate = np.float32(lrate * 0.1)
                                message('Discount learning rate to {} at iteration {} at workder {}'.format(lrate, uidx,
                                                                                                            worker_id))
                            else:
                                clip_shared.set_value(np.float32(clip_shared.get_value() * 0.1))
                                message(
                                    'Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))
                            finetune_cnt += 1
                            if finetune_cnt == 3:
                                message(
                                    'Learning rate decayed to {:.5f}, clip decayed to {:.5f}, task completed'.format(
                                        lrate, clip_shared.get_value()))
                                return 1., 1., 1.
                            bad_counter = 0

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after {} iterations!'.format(uidx)
                estop = True
                break
        print 'Seen {} samples in worker {}'.format(n_samples, worker_id)

        if estop:
            break

    if best_p is not None:
        zipp(best_p, model.P)

    use_noise.set_value(0.)
    value_return['reward'] = best_bleu

    # return reward here
    return value_return


if __name__ == '__main__':
    pass
