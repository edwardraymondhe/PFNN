import numpy as np
import theano
import theano.tensor as T

from pfnn import PhaseFunctionedNetwork, AdamTrainer

theano.config.allow_gc = True
rng = np.random.RandomState(23456)


""" Construct Network """

network = PhaseFunctionedNetwork(rng=rng, dropout=0.7, mode='train')

""" Construct Trainer """

batchsize = 32
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.0001)

""" Start Training """

I = np.arange(len(network.X))

for me in range(20):
    rng.shuffle(I)
    
    print('\n[MacroEpoch] %03i' % me)
    
    for bi in range(10):
    
        """ Find Batch Range """
        
        start, stop = ((bi+0)*len(I))//10, ((bi+1)*len(I))//10
        
        """ Load Data to GPU and train """
        
        E = theano.shared(np.concatenate([network.X[I[start:stop]], network.P[I[start:stop]][...,np.newaxis]], axis=-1), borrow=True)
        F = theano.shared(network.Y[I[start:stop]], borrow=True)
        trainer.train(network, E, F, filename='./demo/network/pfnn/network.npz', restart=False, shuffle=False)
        
        """ Unload Data from GPU """
        
        E.set_value([[]]); del E
        F.set_value([[]]); del F
        
        """ Save Network """
        
        network.save_network()

