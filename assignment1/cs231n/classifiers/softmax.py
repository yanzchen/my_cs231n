import numpy as np
from random import shuffle

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
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    scores = np.dot(X[i], W) # (num_class,)
    scores -= np.max(scores) # take care of numeric stability
    exp_scores = np.exp(scores) 
    probs = exp_scores / np.sum(exp_scores) # (num_class,)
    yi = y[i] # position of correct class on probs vector
    correct_prob = probs[yi]
    loss += -np.log(correct_prob)

    # Our goal is the grads WRT W (dW) from the loss. The final loss 
    # props back via a chain: log, probs, scores and X[i].
    #
    # Although only one prob (i.e. probs[yi]) is included into the loss, 
    # this *single prob* is computed from *all* (X[i].W) scores in that row.
    # Therefore, all scores in that row receive grads from that one-prob loss.
    #
    # It seems rather complicated to formulate the loss-grads WRT to the scores ...
    #
    # Anyway, we update dW with the contributions from X[i] as follow:
    # X[i].W leads to one row of scores, and grads of loss WRT these scores 
    # are determined with this formula: { dscore_j = p_j - 1(j==yi) }
    # 
    # Once we have computed one dscore row, we can backprop in terms of
    # dW[:, j] columns -- whole dW[:, j] column vector participates
    # into score_j.
    dscores = probs
    dscores[yi] -= 1 # postion associated with the correct class
    for j in range(num_class):
        dW[:, j] += dscores[j] * X[i] # make sure to accumulate (for all X[i])!


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  dW /= num_train
  dW += reg * W

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
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  Scores = np.dot(X, W) # (N, C)
  Scores -= np.max(Scores) # for numeric stability when exp them
  Probs = np.exp(Scores) / (np.sum(np.exp(Scores), axis=1)[:, np.newaxis])
  correct_Probs = Probs[range(num_train), y] 
  loss = - np.sum(np.log(correct_Probs))
  loss /= num_train

  # D-Loss/D-score per cell; together they form dScores matrix
  dScores = Probs
  dScores[range(num_train), y] -= 1 # yi rule
  # (Scores = X dot W) ==> (dScores = X dot dW) ==> (dW = X.T dot dScores)
  dW = np.dot(X.T, dScores)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

