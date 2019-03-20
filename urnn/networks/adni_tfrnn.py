import numpy as np

from .tfrnn import TFRNN

class ADNI_TFRNN(TFRNN):
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
        ttVar=None,
        ttRank=None,
        imTTmodes=None,
        hdTTmodes=None,
        viz=None,
        ShowViz=None,
        b_print_rate=10):

        super(ADNI_TFRNN, self).__init__(name=name,
                                    rnn_cell=rnn_cell,
                                    num_in=num_in,
                                    num_hidden=num_hidden, 
                                    num_out=num_out,
                                    num_target=num_target,
                                    single_output=single_output,
                                    activation_hidden=activation_hidden,
                                    activation_out=activation_out,
                                    optimizer=optimizer,
                                    loss_function=loss_function,
                                    seq_len=seq_len,
                                    ttVar=ttVar,
                                    ttRank=ttRank,
                                    imTTmodes=imTTmodes,
                                    hdTTmodes=hdTTmodes,
                                    viz=viz,
                                    ShowViz=ShowViz,
                                    b_print_rate=b_print_rate)

    def validate(self, sess, dataset, epoch_idx, batch_size):
        valid_gt = []
        valid_pd = []
        tmp = 0
        nvalid = 25
        for i in range(0,nvalid):
            X_val, Y_val = dataset.get_validation_datum(i)
            validation_loss, valid_pred = self.evaluate(sess, X_val, Y_val)
            valid_gt.append(Y_val)
            valid_pd.append(valid_pred)
            tmp += validation_loss

            if i==1:
                gttmp = np.concatenate((X_val, np.expand_dims(Y_val,axis=1)), axis=1)
                pdtmp = np.concatenate((X_val, np.expand_dims(valid_pred,axis=1)), axis=1)
                self.viz.updateViz(gt=gttmp, pd=pdtmp)
                vizpath = self.valimgpath+'epoch_'+str(epoch_idx).zfill(2)+'_valid_imgs.png'
                self.viz.saveIt(path=vizpath)
                if self.ShowViz == True:
                    self.viz.showIt()


        validation_loss = tmp/nvalid
        # validation_loss, valid_pred = self.evaluate(sess, X_val, Y_val)
        np.save(file=self.valimgpath+'_valid_gt.npy', arr=valid_gt)
        np.save(file=self.valimgpath+'_valid_pd.npy', arr=valid_pd)

        return validation_loss