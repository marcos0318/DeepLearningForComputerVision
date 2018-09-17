import numpy as np
from random import shuffle
# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(y.shape[0]):
     scores = X[i].dot(W)
     shifted = scores - max(scores)
     loss_i = - shifted[y[i]] + np.log(sum(np.exp(shifted)))
     loss += loss_i
     for j in range(W.shape[1]):
         softmax_output = np.exp(shifted[j])/sum(np.exp(shifted))
         if j == y[i]:
             dW[:,j] += (-1 + softmax_output) *X[i] 
         else: 
             dW[:,j] += softmax_output *X[i] 

  loss /= y.shape[0] 
  loss +=  0.5 * reg * np.sum(W * W)
  dW = dW/y.shape[0] +  reg* W 
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  product = X.dot(W)
  # numerical stability
  max_product = np.max(product, axis=1).reshape([-1, 1])
  product -= max_product
  exp_product = np.exp(product)
  exp_produxt_sum = np.sum(exp_product, axis=1)
  exp_product /= exp_produxt_sum.reshape([-1, 1])
  individual_losses = -np.log(exp_product[np.arange(X.shape[0]), y])
  loss = np.mean(individual_losses) + 0.5 * reg * np.sum(W * W)

  dS = exp_product
  dS[range(y.shape[0]), y] += -1
  dW = X.T.dot(dS)
  dW = dW/y.shape[0] + reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

