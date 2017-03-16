import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    # X[i] dot W[DxC] ==> scores[C] ==> Li (a list of Li_j from C scalar scores)
    # compute gradient contribution from X[i] to W[DxC].

    yi = y[i] # target label position
    #scores = X[i].dot(W)
    scores = np.dot(X[i], W)
    #correct_class_score = scores[yi]
    for j in xrange(num_classes):
      #if j == y[i]:
      if j == yi:
        continue
      #margin = scores[j] - correct_class_score + 1 # note delta = 1
      margin = scores[j] - scores[yi] + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # scores[j] is from (X[i] . W[:, j]), and scores[yi] is from (X[i] . W[:, yi])
        dW[:, j] += X[i]   # column j
        dW[:, yi] -= X[i]  # column yi, same X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # grad
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # grad for regularization loss: reg * W
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  # Plan of vectoring: start Scores matrix, then gradually update the matrix,
  # then at the end sum the matrix for total data loss (then divide by num_trains).

  # Vectorize (Scores - scores_y + 1) first
  Scores = np.dot(X, W) # (N, C) scores -- each row contains C-scores for an X-row

  scores_y = Scores[range(num_train), y]
  scores_y = scores_y[:, np.newaxis] # for broadcast into row-elements

  Losses = Scores - scores_y + 1 # margins

  Losses[Losses<0] = 0 # max(0, margin)
  Losses[range(num_train), y] = 0 # Scores[:, yi] excluded from loss equation

  loss = np.sum(Losses) / num_train
  loss += 0.5 * reg * np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  # Inspired by bruceoutdoors solution (https://github.com/bruceoutdoors/CS231n):
  #
  # We build a loss-map on the Scores matrix. Each cell on this loss-map represents
  # the number of loss-contributing occurrances from the score at that cell position.
  # For every loss-contributing occurrance at cell [i,j], gradient of X[i] amount will
  # be assessed to dW[:, j] (one X[i] row and one W-column computes into one Score[i, j]).
  # Loss-map = dot(X, W) ==> dW = dot(X.T, Loss-map)
  Loss_map = np.zeros(Losses.shape)
  Loss_map[Losses>0] = 1 # every regular cell only contributes to at most 1 loss 
  
  # Target (yi) cell may be associated with multiple negative losses:
  # for each regular loss assessed on a Scores row, a negative-loss
  # is assessed on the target cell.
  Loss_sum = np.sum(Loss_map, axis=1)
  Loss_map[range(num_train), y] = -Loss_sum

  dW = np.dot(X.T, Loss_map)
  
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
