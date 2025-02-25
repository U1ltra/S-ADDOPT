########################################################################################################################
####-------------------------------------Logistic Regression for MNIST dataset--------------------------------------####
########################################################################################################################

## Used to help implement decentralized algorithms to classify MNIST dataset

import numpy as np
from numpy import linalg as LA
import os
import sys
import copy as cp


class LR_L2( object ):
    def __init__(self, n_agent, class1 = 2, class2 = 6, train = 12000, balanced = True, limited_labels = False ):
        self.class1 = class1
        self.class2 = class2
        self.train = train
        self.limited_labels = limited_labels
        self.n = n_agent 
        self.balanced = balanced
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_data()
        print(f"Data size {len(self.X_train)}")

        self.N = len(self.X_train)            ## total number of data samples
        if balanced == False:
            self.split_vec = np.sort(
                np.random.choice(
                    np.arange(1, self.N), self.n-1, replace = False 
                )
        ) 
            
        self.noniid = True

        self.X, self.Y, self.data_distr = self.distribute_data()
        for i, dataset in enumerate(self.X):
            print(f"client {i} {len(dataset)}")
            
        self.p = len(self.X_train[0])         ## dimension of the feature 
        print(f"feat dim {self.p }")

        self.reg = 0.2 / 2
        self.dim = self.p                     ## dimension of the feature 
        self.L, self.kappa = self.smooth_scvx_parameters()
        self.b = int(self.N/self.n)           ## average local samples
        print(f"L-smooth constant {self.L}")

    def load_data(self):
        if os.path.exists('mnist.npz'):
            print( 'data exists' )
            data = np.load('mnist.npz', allow_pickle=True)
            X = data['X']
            y = data['y']
        else:
            print( 'downloading data' )
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            np.savez_compressed('mnist', X=X, y=y)
        y = y.astype(int)
        print( 'data initialized' )

        ## append 1 to the end of all data points
        X = np.append(X, np.ones((X.shape[0],1)), axis = 1)
        
        ## data normalization: each data is normalized as a unit vector 
        X = X / LA.norm(X,axis = 1)[:,None]
        
        ## select corresponding classes
        X_C1_C2 = X[ (y == self.class1) | (y == self.class2) ]
        y_C1_C2 = y[ (y == self.class1) | (y == self.class2) ]
        y_C1_C2[ y_C1_C2 == self.class1 ] = 1    
        y_C1_C2[ y_C1_C2 == self.class2 ] = -1
        X_train, X_test = X_C1_C2[ : self.train], X_C1_C2[ self.train : ]
        Y_train, Y_test = y_C1_C2[ : self.train], y_C1_C2[ self.train : ]
             
        if self.limited_labels == True:
            permutation = np.argsort(Y_train)
            X_train = X_train[permutation]
            Y_train = np.sort(Y_train)
            
        return X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy() 
    
    def distribute_data(self):
        if self.balanced == True:
           if self.noniid:
               idx = np.argsort(self.Y_train)
               self.X_train = self.X_train[idx]
               self.Y_train = self.Y_train[idx]
            
           X = np.array( np.split( self.X_train, self.n, axis = 0 ) ) 
           Y = np.array( np.split( self.Y_train, self.n, axis = 0 ) ) 
        if self.balanced == False:   ## random distribution
           X = np.array( np.split(self.X_train, self.split_vec, axis = 0) )
           Y = np.array( np.split(self.Y_train, self.split_vec, axis = 0 ) )
        data_distribution = np.array([ len(_) for _ in X ])
        return X, Y, data_distribution
    
    def smooth_scvx_parameters(self):
        Q = np.matmul(self.X_train.T,self.X_train)/self.N
        L_F = max(abs(LA.eigvals(Q)))/4
        L = L_F + self.reg
        kappa = L/self.reg
        return L, kappa
    
    def F_val(self, theta):           ##  objective function value at theta
        if self.balanced == True:
            f_val = np.sum( np.log( np.exp( np.multiply(-self.Y_train,\
                                                    np.matmul(self.X_train,theta)) ) + 1 ) )/self.N
            reg_val = (self.reg/2) * (LA.norm(theta) ** 2) 
            return f_val + reg_val
        if self.balanced == False:
            temp1 = np.log( np.exp( np.multiply(-self.Y_train,\
                              np.matmul(self.X_train,theta)) ) + 1 ) 
            temp2 = np.split(temp1, self.split_vec)
            f_val = 0
            for i in range(self.n):
                f_val += np.sum(temp2[i])/self.data_distr[i]
            reg_val = (self.reg/2) * (LA.norm(theta) ** 2) 
            return f_val/self.n + reg_val
            
        
    def localgrad(self, theta, idx, j = None, permute = None, permute_flag = False):  ## idx is the node index, j is local sample index
        """
            Simulate the local gradient computation at each node. 

            @param
            :theta          current parameter set of the model (Global Model)
            :idx            node index
            :j              local one-sample index
            :permute        the set of samples that will be used for the local 
                            gradient computation of node idx
            :permute_flag   if permute_flag == True, then permute is not None
        """
        if permute_flag:
            assert j == None
            temp1 = np.exp( 
                np.matmul(
                    self.X[idx][permute], theta[idx]) * (-self.Y[idx][permute])  
            )
            temp2 = ( temp1/(temp1+1) ) * (-self.Y[idx][permute])
            grad = self.X[idx][permute] * temp2[:,np.newaxis]

            return np.sum(grad, axis = 0) / len(permute) + self.reg * theta[idx]
    
        
        if j == None:                 ## local full batch gradient
            temp1 = np.exp( np.matmul(self.X[idx],theta[idx]) * (-self.Y[idx])  )
            temp2 = ( temp1/(temp1+1) ) * (-self.Y[idx])
            grad = self.X[idx] * temp2[:,np.newaxis]
            return np.sum(grad, axis = 0)/self.data_distr[idx] + self.reg * theta[idx]
        else:                         ## local stochastic gradient  
            temp = np.exp(self.Y[idx][j]*np.inner(self.X[idx][j], theta[idx]))
            grad_lr = -self.Y[idx][j]/(1+temp) * self.X[idx][j]                         # TODO: I highly suspect they are missing a <code>temp</code> here. But since sigmoid's range is [0,1]. Missing might not cause a big problem
            grad_reg = self.reg * theta[idx]
            grad = grad_lr + grad_reg
            return grad
        
    def networkgrad(self, theta, idxv = None, permute = None, permute_flag = None):  ## network stochastic/batch/mini-batch gradient
        """
            Optimizer for DSGD and DRR. All graph network based optimizer will 
            call this function.

            @param
            :theta          current parameter set of the model (Global model)
            :idxv           index vector for one-sample stochastic gradient on each nodes
            :permute        a set of samples for gradient computation
            :permute_flag   whether to use our implementation of gradient computation

            @return
            :grad           gradient of the objective function at each node
        """
        grad = np.zeros( (self.n,self.p) )

        if permute_flag:
            for i in range(self.n):
                grad[i] = self.localgrad(theta , i, permute = permute[i], permute_flag = True)
            return grad
        
        elif idxv is None:                        ## full batch
            for i in range(self.n):
                grad[i] = self.localgrad(theta , i)
            return grad
        else:                                   ## stochastic gradient: one sample
            for i in range(self.n):
                grad[i] = self.localgrad(theta, i, idxv[i])
            return grad
    
    def grad(self, theta, idx = None, permute = None, permute_flag = None): ## centralized stochastic/batch gradient
        """ 
            Gradient Computation for CSGD and CRR. Note that in our experiment, 
            only permute and permute_flag are useful parameters

            @param
            :theta          current parameter set of the model
            :idx            index of the sample to be used for gradient computation (for stochastic gradient)
            :permute        a set of samples to be used for gradient computation (for CSGD and CRR)
            :permute_flag   a flag to indicate whether to use our implementations of CSGD or CRR

            @return
            :grad           averaged gradient of the objective function at theta on the given set of samples
        """
        if permute_flag:
            # Both SGD & RR is implemented here
            # SGD will randomly permute all indices and pass in the first batch_size number of indices
            # CRR will ensure that the entire permutation set has been looked through before the next permutation
            temp1 = np.exp( 
                np.matmul(self.X_train[permute], theta) * (-self.Y_train[permute])  
            )
            temp2 = ( temp1/(temp1+1) ) * (-self.Y_train[permute])
            grad = self.X_train[permute] * temp2[:,np.newaxis]

            return np.sum(grad, axis = 0) / len(permute) + self.reg * theta

        if idx == None:                ## full batch
            if self.balanced == True:
                temp1 = np.exp( np.matmul(self.X_train,theta) * (-self.Y_train)  )
                temp2 = ( temp1/(temp1+1) ) * (-self.Y_train)
                grad = self.X_train * temp2[:,np.newaxis]
                return np.sum(grad, axis = 0)/self.N + self.reg * theta
            
            if self.balanced == False:                                          # TODO: how could contralized gradient be imbalanced??？
                return np.sum( self.networkgrad(np.tile(theta,(self.n,1)))\
                              , axis = 0 )/self.n
        else:
            if self.balanced == True:
                temp = np.exp(self.Y_train[idx]*np.inner(self.X_train[idx], theta))
                grad_lr = -self.Y_train[idx]/(1+temp) * self.X_train[idx]
                grad_reg = self.reg * theta
                grad = grad_lr + grad_reg
                return grad
            if self.balanced == False:
                sys.exit( 'data distribution is not balanced !!!' )