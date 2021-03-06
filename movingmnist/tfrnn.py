import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt

import datetime as dt
import time

def serialize_to_file(file, losses):
    file=open(file, 'w')
    for l in losses:
        file.write("{0}\n".format(l))
    file.close()

class TFRNN:
    def __init__(
        self,
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function,
        seq_len,
        # ttVar=None,
        ttVar1=None,
        ttVar2=None,
        ttRank=None,
        imTTmodes=None,
        hdTTmodes=None,
        viz=None,
        ShowViz=None,
        b_print_rate=10):

        self.b_print_rate = b_print_rate
        self.viz = viz
        self.ShowViz = ShowViz
        self.single_output = single_output

        # self
        self.name = name
        self.loss_list = []
        self.init_state_C = np.sqrt(3 / (2 * num_hidden))
        self.log_dir = './logs/'
        # self.writer = tf.summary.FileWriter(self.log_dir)
        self.runTime = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.runName = self.name + '_' + self.runTime + '/'
        self.chkpath = 'chkpts/' + self.runName
        self.respath = 'res/' + self.runName
        self.valimgpath = self.respath + 'valid/'

        # init cell
        print('Initializing Cell...')
        # init cell
        if ttRank is not None:
            self.cell = rnn_cell(input_shape = imTTmodes,
                                 hidden_shape = hdTTmodes,
                                 output_shape = imTTmodes,
                                 ttRank=ttRank,
                                 # ttvar=ttVar,
                                 ttvar1=ttVar1, ttvar2=ttVar2,
                                 activation = activation_hidden)
        else:
            self.cell = rnn_cell(input_shape = imTTmodes,
                                 hidden_shape = hdTTmodes,
                                 output_shape = imTTmodes,
                                 activation = activation_hidden)

        # extract output size
        self.output_size = self.cell.output_size
        if type(self.output_size) == dict:
            self.output_size = self.output_size['num_units']
      
        # input_x: [batch_size, max_time, num_in]
        # input_y: [batch_size, max_time, num_target] or [batch_size, num_target]
        self.input_x = tf.placeholder(tf.float32, [None, None, num_in], name="input_x_"+self.name)
        self.input_y = tf.placeholder(tf.float32, [None, num_target] if single_output else [None, None, num_target],  
                                      name="input_y_"+self.name)
        
        # rnn initial state(s)
        self.init_states = []
        self.dyn_rnn_init_states = None

        # prepare state size list
        # if type(self.cell.state_size) == int:
        #     state_size_list = [self.cell.state_size]
        #     self.dyn_rnn_init_states = self.cell.state_size # prepare init state for dyn_rnn
        # elif type(self.cell.state_size) == tf.contrib.rnn.LSTMStateTuple:
        #     state_size_list = list(self.cell.state_size)
        #     self.dyn_rnn_init_states = self.cell.state_size # prepare init state for dyn_rnn

        # construct placeholder list==
        # for state_size in state_size_list:
        #     init_state = tf.placeholder(tf.float32, [None, state_size], name="init_state")
        #     self.init_states.append(init_state)
        
        # # prepare init state for dyn_rnn
        # if type(self.cell.state_size) == int:
        #     self.dyn_rnn_init_states = self.init_states[0]
        # elif type(self.cell.state_size) == tf.contrib.rnn.LSTMStateTuple:
        #     self.dyn_rnn_init_states = tf.contrib.rnn.LSTMStateTuple(self.init_states[0], self.init_states[1])

        self.init_state = tf.placeholder(tf.float32, [None, num_hidden], name="init_state")
        # self.init_state = tf.placeholder(tf.float32, [None]+hdTTmodes, name="init_state")
        self.dyn_rnn_init_states = self.init_state

        print("Initializing RNN...")
        # run the dynamic rnn and get hidden layer outputs
        # outputs_h: [batch_size, max_time, self.output_size]
        outputs_h, final_state = tf.nn.dynamic_rnn(self.cell, self.input_x, initial_state=self.dyn_rnn_init_states) 

        print("Initializing Loss and Optimizers...")
        if single_output:
            outputs_o = outputs_h[:, -1 , :]
        else:
            outputs_o = outputs_h
            
        # self.predictions = tf.reshape(outputs_o, [-1, 121*145*121], name="thebigreshape")
        self.predictions = activation_out(outputs_o)
        # calculate losses and set up optimizer

        # loss function is usually one of these two:
        #   tf.nn.sparse_softmax_cross_entropy_with_logits 
        #     (classification, num_out = num_classes, num_target = 1)
        #   tf.squared_difference 
        #     (regression, num_out = num_target)
        if loss_function == tf.squared_difference:
            self.total_loss = tf.reduce_mean(loss_function(self.predictions, self.input_y))
        elif loss_function == tf.nn.sparse_softmax_cross_entropy_with_logits:
            prepared_labels = tf.cast(tf.squeeze(self.input_y), tf.int32)
            self.total_loss = tf.reduce_mean(loss_function(logits=outputs_o, labels=prepared_labels))
        elif loss_function == tf.nn.sigmoid_cross_entropy_with_logits:
            prepared_labels = tf.round(self.input_y)
            self.total_loss = tf.reduce_mean(loss_function(logits=outputs_o, labels=prepared_labels))
        else:
            raise Exception('Unknown loss function')
        # if rnn_cell == OTTRNNCell:
        #     EucGnVs = optimizer.compute_gradients(self.total_loss, [self.w_ho, self.b_o, self.cell.w_ih, self.cell.b_h])
        #     myEucgrads = [(g, v) for g, v in EucGnVs]
        #     self.E_train_step = optimizer.apply_gradients(myEucgrads)

        #     # lr = 1e-4
        #     AottGnVs = optimizer.compute_gradients(self.total_loss, [self.cell.W1.getQ()])#getQ())
        #     self.myAottgrads = [(g, v) for g, v in AottGnVs]
        #     self.S_train_step = optimizer.apply_gradients(self.myAottgrads)
        #     # self.S_train_step = [v.assign(gradStep(v, g, lr)) for g, v in AottGnVs]
        # else:    
        self.train_step = optimizer.minimize(self.total_loss, name='Optimizer')

        # checkpointing
        self.saver = tf.train.Saver(max_to_keep=100)

        # only make the directories if the model inits correctly.
        os.makedirs(self.respath)
        os.makedirs(self.valimgpath)
        os.makedirs(self.chkpath)

        # tensorboard
        # self.writer.add_graph(tf.get_default_graph())
        # self.writer.flush()
        # self.writer.close()

        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Network __init__ over. Number of trainable params=', t_params)
        if t_params > 1000000000:
            print('Network probably too large for a single GPU, exiting.')
            print('Check network architecture to make sure it fits.')
            exit(0)

    def train(self, dataset, batch_size, epochs):           

        # session
        # config = tf.ConfigProto()
        # with tf.Session(config=config) as sess:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # initialize global vars
            sess.run(tf.global_variables_initializer())

            # fetch validation and test sets
            num_batches = dataset.get_batch_count(batch_size)

            # init loss list
            self.loss_list = []
            self.vloss_list = []
            self.mbloss_list = []
            print("Starting training for", self.name)
            print("NumEpochs:", '{0:3d}'.format(epochs), 
                  "|BatchSize:", '{0:3d}'.format(batch_size), 
                  "|NumBatches:", '{0:5d}'.format(num_batches),'\n')


            # train for several epochs
            for epoch_idx in range(epochs):

                t0 = time.time()
                print("Epoch Starting:", epoch_idx, '\n')
                # train on several minibatches
                for batch_idx in range(num_batches):

                    # get one batch of data
                    # X_batch: [batch_size x time x num_in]
                    # Y_batch: [batch_size x time x num_target] or [batch_size x num_target] (single_output?)
                    X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)

                    # evaluate
                    batch_loss, _ = self.evaluate(sess, X_batch, Y_batch, training=True)
                    self.loss_list.append(batch_loss)

                    serialize_to_file(file= self.respath+'batch_losses.txt', losses=self.loss_list)

                    # plot
                    if batch_idx % self.b_print_rate == 0:
                        total_examples = batch_size * num_batches * epoch_idx + batch_size * batch_idx + batch_size

                        # print stats
                        print("Epoch:", '{0:3d}'.format(epoch_idx), 
                              "|Batch:", '{0:3d}'.format(batch_idx), 
                              "|TotalExamples:", '{0:5d}'.format(total_examples), # total training examples
                              "|BatchLoss:", '{0:8.4f}'.format(batch_loss))

                # validate after each epoch
                validation_loss = self.validate(sess, dataset, epoch_idx, batch_size)
                self.vloss_list.append(validation_loss)
                serialize_to_file(file= self.respath+'valid_losses.txt', losses=self.vloss_list)

                mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
                self.mbloss_list.append(mean_epoch_loss)
                print("Epoch Over:", '{0:3d}'.format(epoch_idx), 
                      "|MeanEpochLoss:", '{0:8.4f}'.format(mean_epoch_loss),
                      "|ValidationSetLoss:", '{0:8.4f}'.format(validation_loss),'\n')
                serialize_to_file(file= self.respath+'meanbatch_losses.txt', losses=self.mbloss_list)

                t1 = time.time()
                myT = t1 - t0
                print('Epoch over, took', myT, 'seconds.')
                if (epoch_idx+1) % 25 == 0:
                    print('Saving checkpoint,',epoch_idx,' epoch.')
                    # save intermediate
                    self.saver.save(sess, self.chkpath+'ep_'+str(epoch_idx).zfill(2)+'.ckpt')


    def test(self, dataset, batch_size, epochs):
        # session
        with tf.Session() as sess:
            # initialize global vars
            sess.run(tf.global_variables_initializer())

            # fetch validation and test sets
            X_test, Y_test = dataset.get_test_data()

            test_loss = self.evaluate(sess, X_test, Y_test)
            print("Test set loss:", test_loss)

    def evaluate(self, sess, X, Y, training=False):

        # fill (X,Y) placeholders
        feed_dict = {self.input_x: X, self.input_y: Y}
        batch_size = X.shape[0]

        # fill initial state
        # for init_state in self.init_states:
            # init_state: [batch_size x cell.state_size[i]]
        feed_dict[self.init_state] = np.random.uniform(-self.init_state_C, self.init_state_C, [batch_size,self.cell.state_size])
        # feed_dict[self.init_state] = np.random.uniform(-self.init_state_C, self.init_state_C, [batch_size]+self.cell.state_shape)

        # run and return the loss
        if training:
            loss, _ = sess.run([self.total_loss, self.train_step], feed_dict)
            pred = None
        else:
            loss, pred = sess.run([self.total_loss, self.predictions], feed_dict)

        return loss, pred

    def validate(self, sess, dataset, epoch_idx, batch_size):
        # fetch validation and test sets
        X_val, Y_val = dataset.get_validation_data(batch_size)

        validation_loss, valid_pred = self.evaluate(sess, X_val, Y_val)

        if self.single_output:
            gttmp = np.concatenate((X_val, np.expand_dims(Y_val,axis=1)), axis=1)
            pdtmp = np.concatenate((X_val, np.expand_dims(valid_pred,axis=1)), axis=1)
        else:
            gttmp = X_val
            pdtmp = valid_pred

        self.viz.updateViz(gt=gttmp, pd=pdtmp)
        self.viz.saveIt(self.valimgpath+'epoch_'+str(epoch_idx).zfill(2)+'_valid_imgs.png')
        if self.ShowViz == True:
            self.viz.showIt()


        np.save(file=self.valimgpath+'_valid_gt.npy', arr=Y_val)
        np.save(file=self.valimgpath+'_valid_pred.npy', arr=valid_pred)

        return validation_loss
