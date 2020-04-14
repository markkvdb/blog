---
layout: post
description: This post dicusses one of the fundamental building blocks of machine learning.
categories: [R, Machine Learning]
comments: true
---

# Hidden Markov Model Tutorial in R

This document contains an introduction to Hidden Markov Models (HMMs).
First, a brief description and the main problems of HMMs will discussed.
After, I will provide common strategies to analyse these problems.
Lastly, I apply the HMM framework on a speech recognition problem.

## Model Formulation

A HMM models a Markov process which affects some observerable
process(es). Any HMM model can be defined with 5 elements, namely:

1. The set of $$N$$ hidden states $$V = \{v_1,\dots, v_N\}$$;
2. The transition matrix $$Q$$ where the $$i,j$$ -th element represents the transition probability of going from
hidden state $$x_i$$ to $$x_j$$;
3. A sequence of $$T$$ observations $$Y =\{y_1,\dots, y_T\}$$, each drawn from observation set $$D =\{d_1,\dots, d_d\}$$;
4. Functions $$b_i(y_t)$$ that contain the probability of particular observation at time $$t$$ given that the process is in state $$v_i$$. The entire set of functions is denoted by $$B =\{b_j(\cdot):\forall j\in [N]\}$$;
5. The initial hidden state probabilities for time $$t=0$$: $$\pi = [\pi_1,\dots,\pi_N]$$.

We indicate $$\lambda$$ as a short-hand notation for the complete set of HMM parameters, i.e., $$\lambda = (Q, B,\pi)$$. The three main problems associated with HMMs are:

1. Find $$P(Y\mid\lambda)$$ for some observation sequence $$Y = (y_1,\dots, y_T)$$.
2. Given some $$Y$$ and $$\lambda$$, find the best (hidden) state sequence $$X = (x_1,\dots, x_T)$$.
3. Find the HMM parameters that maximises $$P(Y\mid\lambda)$$, i.e., find $$\lambda^* =\text{argmax}_{\lambda}P(Y\mid\lambda)$$.

In the remainder of this article I will provide approaches to solve each
of these problems and provide an implementation in R. Before discussing
an interesting application for HMMs, I will provide a very simple HMM to
discuss the three main problems for clarity.

This simple HMM example in R is given below:

    # Define model
    V = c("HOT", "COLD")
    Q = matrix(c(0.7, 0.3, 0.4, 0.6), nrow=2, byrow=TRUE)
    D = c(1, 2, 3)
    Y = c(1, 3, 2, 3)
    B = matrix(c(0.2, 0.4, 0.4, 0.6, 0.3, 0.1), nrow=3)
    pi = c(0.5, 0.5)

## Forward probabilities

One interesting problem for HMMs is determining the likelihood of a
given sequence of observations given the HMM parameters $$\lambda$$. As opposed to regular Markov models, this is not straight-forward to
compute, since we do not know the underlying hidden state sequence.

One possible solution would be to compute the likelihood of a given
observation sequence by all possible hidden state sequences that support
this observation sequene. In our toy model, the observation space does
not depend on the hidden state, hence all sequences of hidden states
have to be considered.

This method is commonly referred to as the law of total expectations or
Tower rule. The idea is that we compute $$P(Y)$$ by using:

$$
  P(Y\mid\lambda) =\sum_{x\in X} P(Y\mid x,\lambda)P(x\mid\lambda),
$$

where $$X$$ is the set of all valid hidden state sequences, e.g., $$X =\{V^T\}$$. We can compute the conditional probability of an observation sequence
given the hidden state sequence as

$$
  P(Y\mid x,\lambda) =\prod_{t=1}^T b_{x_t}(y_t).
$$

Below, I have the R code that computes the likelihood by brute-force:

    # Transform hidden state set to numerical set
    V_num = seq(1, length(V))

    # All possible hidden state sequences 2^12
    V_all = permutations(n=length(V), r=length(Y), repeats.allowed=TRUE)

    # Compute the likelihood given a hidden state sequence
    get_likelihood = function(V_seq, Y, B, pi, Q) {
      l1 = prod(B[matrix(c(Y, V_seq), ncol=2)])

      # Compute all transition probabilitoes
      Q_el = matrix(c(V_seq[1:(length(V_seq)-1)], V_seq[2:length(V_seq)]), ncol=2)
      l2 = pi[V_seq[1]] * prod(Q[Q_el])

      return(l1 * l2)
    }

    total_l = sum(apply(V_all, 1, get_likelihood, Y, B, pi, Q))
    print(total_l)

    ## [1] 0.0099748

This brute-force algorithm is extremely inefficient and is not
applicable when the state space and/or sequence length is large. Just
like (regular) Markov models we can use the Markov property to compute
the likelihood in a more efficient manner. Here, we make use of the fact
that the transition probabilities of jumping to certain states only
depends on the current state. Let $$Y_t$$ denote the subset of $$Y$$ of the first $$t$$ observations and let $$X_t$$ be the set of hidden state sequences up to time $$t$$. Recall that we want to compute $$P(Y)$$. Let $$\alpha_t(j)$$ represent the probability of being in state $$j$$ at time $$t$$ after seeing the first $$t$$ observations $$Y_t$$, given our model specification, i.e.,
*α*<sub>*t*</sub>(*j*) = *P*(*O*<sub>*t*</sub>, *x*<sub>*t*</sub> = *j* ∣ *λ*).
 By the law of total expectation we have that

We can use this recursively relationship to efficiently compute $$P(Y_T\mid\lambda)$$ by means of the which is presented in Alg. (1).

The R-implementation can be found below

    forward_alg = function(Y, V, pi, B, Q) {
      # Define empty forward matrix
      forward = matrix(data=0, nrow=length(V), ncol=length(Y))

      # Fill first elements
      forward[,1] = B[Y[1],] * pi

      # Now for all other time steps
      for (t in 2:length(Y)) {
    forward[,t] = forward[,t-1] %*% Q * B[Y[t],]
      }

      return(forward)
    }

    alpha = forward_alg(Y, V_num, pi, B, Q)
    print(sum(alpha[, length(Y)]))

    ## [1] 0.0099748

We see that the brute-force method and the forward algorithm produce the
same likelihood for our sequence.

## Decoding Hidden States

The forward algorithm can be used to determine the likelihood of a
certain observed sequence given the Markov model and the hidden state
sequence. Note, however, that the hidden state sequence is not observed!
It is more interesting to compute the most likely hidden state sequence
given the underlying Markov model and the observed state sequence. This
task of determining the hidden state sequence is refered to as the.

One possible way to determine the hidden state sequence would be to
compute the likelihood of all possible hidden state sequences using the
forward algorithm and select the hidden state sequence with the highest
likelihood value. Similar to determining the likelihood, this
brute-force approach quickly becomes intractable. Instead, we can apply
the dynamic programming algorithm called the to decode the hidden state
sequence.

Similar to the forward algorithm, the proceeds through the time-series
from the start till the end. The can be computed as

$$
  (x^*_1,\dots, x^*_T) =\argmax_{(x_1,\dots, x_T)\in X} P(Y_T, X_T\mid\lambda) P(X_T\mid\lambda).
$$

Similar to the forward algorithm, we can aply a dynamic programming
approach to compute the. Let $$v_t(j)$$ be the probability of observing sequence $$Y_t$$ using the hidden sequence $$(x^*_1,\dots, x^*_{t-1})$$, that is,

The psuedo-code for the is given in Alg 2.

The R implementation of the Viterbi algorithm is given below:

    viterbi_alg = function(Y, V, pi, B, Q) {
      # Define empty forward matrix
      T_ = length(Y)
      viterbi = matrix(data=0, nrow=length(V), ncol=length(Y))
      path = matrix(data=0, nrow=length(V), ncol=length(Y))

      # Fill first elements
      viterbi[,1] = B[Y[1],] * pi
      path[,1] = rep(0, length(V))

      # Now for all other time steps
      for (t in 2:length(Y)) {
    tmp_val = t(viterbi[,t-1] * Q)
    max_x = max.col(tmp_val)
    viterbi[,t] = tmp_val[,max_x][, 1] * B[Y[t],]
    path[,t] = max_x
      }

      best_path_prob = max(viterbi[,T_])
      best_path_end = which.max(viterbi[,T_])

      # Best path
      best_path = rep(-1, T_)
      best_path[T_] = best_path_end
      for (t in (T_-1):1) {
    best_path[t] = path[best_path[t+1],t+1]
      }

      return(list(best_path_prob, best_path))
    }

    viterbi_results = viterbi_alg(Y, V_num, pi, B, Q)
    print(viterbi_results[[1]])

    ## [1] 0.0037632

    print(V[viterbi_results[[2]]])

    ## [1] "COLD" "HOT"  "HOT"  "HOT"

## Determining the Optimal HMM Parameters

The standard algorithm to estimate the optimal HMM parameters is the
Baum-Welch (BW) algorithm which is a special case of the
Expectation-Maximisation algorithm. The BW algorithm iteratively updates
the HMM parameters and converges to the optimal HMM parameters under
mild convergence conditions.

Before discussing the BW algorithm we need some useful probabilities.
First, similar to the forward probabilities $$\alpha_t(j) = P(y_1,\dots, y_t, x_t = j\mid\lambda)$$ we can compute the backward probabilities $$\beta_t(j) = P(y_{t+1},\dots, y_T\mid x_t = j,\lambda)$$, i.e., given our HMM parameters $$\lambda$$ and that the hidden state at time $$t$$ equals $$j$$ what is the probability that we observe the sequence $$y_{t+1},\dots, y_T$$. Similar to the forward probabilities we can determine the backward
probabilities using dynamic programming as:
The R implementation is given below:

    back_prob_alg = function(Y, V, pi, B, Q) {
    # Define empty forward matrix
      back_prob = matrix(data=0, nrow=length(V), ncol=length(Y))
      T_max = length(Y)

      # Fill first elements
      back_prob[, T_max] = rep(1, length(V))

      # Now for all other time steps
      for (t in (T_max-1):1) {
    back_prob[,t] = Q %*% (back_prob[, t+1] * B[Y[t+1], ])
      }

      return(back_prob)
    }

    beta = back_prob_alg(Y, V_num, pi, B, Q)
    probY = sum(pi * B[Y[1], ] * beta[, 1])
    print(probY)

    ## [1] 0.0099748

Recall: we are interesting in estimating $$Q$$, $$B$$ and $$\pi$$ given our observation sequence $$Y$$. As we will see later on, estimating these quantities typically involve
estimating the frequency of being in a certain state and/or counting the
expected number of transitions from one state to another. More
precisely, we estimate the transition probability from state $$i$$ to $$j$$ as

$$
\hat{q}_{i,j} =\frac{\textrm{expected number of transitions from state } i \textrm{ to } j}{\textrm{expected number of transitions from state } i}.
$$

To estimate $$b_i(y_t)$$, i.e., the probability of observing $$y_t$$ in state $$X_t = i$$, we compute:

$$
\hat{b}_i(y_t) =\frac{\textrm{expected number of times in state  } i \textrm{ while observing } y_t}{\textrm{expected number of times in state } i}.
$$

The initial state probabilities $$\pi$$ will follow directly from quantities computed in the EW algorithm.

We now define

$$
\gamma_t(j) = P(X_t = j\mid Y,\lambda) =\frac{P(Y, X_t = j\mid\lambda)}{P(Y\mid\lambda)}
$$

which is the probability of being in state $$j$$ at time $$t$$ for a state sequence $$Y$$. By the Markovian conditional independence

$$
\alpha_t(j)\beta_t(j) = P(y_1,\dots, y_t, X_t = j\mid\lambda) P(y_{t+1},\dots, y_T\mid X_t = j,\lambda) = P(Y, X_t = j\mid\lambda).
$$

Hence, we can write $$\gamma_t(j)$$ in terms of $$\alpha_t(j)$$ and $$\beta_t(j)$$ as

$$
\gamma_t(j) =\frac{\alpha_t(j)\beta_t(j)}{P(Y\mid\lambda)}.
$$

Recall that $$P(Y\mid\lambda)$$ can be easily obtained when computing the forward (or backward)
probabilities.

    ComputeGamma <- function(alpha, beta, probY) {
      # Obtain gamma for all t and v
      gamma <- (alpha * beta) / probY
    }

    gamma = ComputeGamma(alpha, beta, probY)

We can now estimate the elements of $$B$$ as

$$
\hat{b}_j(k) =\frac{\sum_{t=1}^T\delta_{y_t, v_k}\gamma_t(j)}{\sum_{t=1}^T\gamma_t(j)},
$$

where $$\delta_{i, j}$$ evaluates to 1 if $$i = j$$ and 0 otherwise.

    Delta <- function(A, b) {
      return(as.integer(A == b))
    }
    Delta = Vectorize(Delta, "b")

    ComputeBHat <- function(gamma, Y, D) {
      # Obtain the delta variable
      delta <- t(Delta(Y, D))

      # Compute nominator and denominator
      deltaGamma <- delta %*% t(gamma)
      deltaDenom <- matrix(data=1, nrow=nrow(delta), ncol=ncol(delta)) %*% t(gamma)

      # Divide element-wise
      BHat <- deltaGamma / deltaDenom

      return(BHat)
    }

    BHat = ComputeBHat(gamma, Y, D)

To estimate the elements of $$Q$$ we define

$$
\psi_t(i,j) = P(X_t = i, X_{t+1} = j\mid Y,\lambda) =\frac{P(X_t = i, X_{t+1} = j, Y\mid\lambda)}{P(Y\mid\lambda)}
$$

which is the probability of being in state $$i$$ at time $$t$$ and being in state $$j$$ at time $$t+1$$ for a state sequence $$Y$$. This can be parameterised in terms of $$\alpha_t(j)$$ and $$\beta_t(j)$$ as

$$
\psi_t(i,j) =\frac{\alpha_t(i)q_{i,j}b_{t+1}(j)\beta_{t+1}(j)}{P(Y\mid\lambda)}.
$$

    ComputePsi <- function(alpha, beta, B, Q, Y, probY) {
      # Create empty matrix
      T_max = ncol(beta)
      psi = array(-1, dim=c(T_max-1, nrow(Q), nrow(Q)))

      for (t in 1:(T_max-1)) {
    psi[t,, ] = cbind(alpha[, t], alpha[, t]) * Q * rbind(B[Y[t+1], ], B[Y[t+1], ]) * rbind(beta[, t+1], beta[, t+1])
      }

      return(psi / probY)
    }

    psi = ComputePsi(alpha, beta, B, Q, Y, probY)

The transition probabilities can now be estimated as

$$
\bar{q}_{i,j} =\frac{\sum_{t = 1}^{T-1}\psi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}
$$

since the expected number of times in a state $$j$$ is equal to the expected number of transitions from state $$j$$ (for an ergodic Markov process).

    ComputeQ = function(psi, gamma) {
      QNom = apply(psi, c(2,3), sum)
      QDenom = t(rep(1, 2)) %x% apply(gamma[, 1:(ncol(gamma)-1)], 1, sum)

      Q = QNom / QDenom
    }

    Q_new = ComputeQ(psi, gamma)

The quantity

$$
\bar{\pi}_j =\gamma_1(j)
$$

is an estimate for the initial state probability for state $$j$$.

The idea of the BW algorithm is to iteratively update $$\gamma_t(\cdot)$$ and $$psi_t(\cdot)$$ in one step (E-step) and the estimates for $$Q$$ and $$B$$ in the other step (M-step). The exact BW algorithm is provided in Alg.
3.

The R implementation of the Balm-Welsch algorithm is given below:

    balm_welch_alg = function(Y, Q, B, pi, V, D) {
      converged = FALSE
      max_iter = 1000
      iter = 1

      while(!converged & iter!= max_iter) {
    # First compute alpa and beta
    alpha = forward_alg(Y, V, pi, B, Q)
    beta = back_prob_alg(Y, V, pi, B, Q)
    probY = sum(alpha[,ncol(alpha)])

    # E-Step
    gamma = ComputeGamma(alpha, beta, probY)
    psi = ComputePsi(alpha, beta, B, Q, Y, probY)

    # M-Step
    Q = ComputeQ(psi, gamma)
    B = ComputeBHat(gamma, Y, D)
    pi = gamma[, 1]

    # Increase iteration
    iter = iter + 1
      }
    }

We have now addressed all main problems concerning HMMs. Note, however
that our implementation in R is rather simplistic and probably has
several numerical issues. Furthermore, we restricted our observed
sequence to consist of (finite) discrete options. A common extension is
to allow $$Y\mid X$$ to have some continous distribution which depends on $$X$$ and is parameterised by some $$\theta$$. Check Gaussian Mixture Hidden Markov Models as an example.

## Numerical Issues

The procedures described in this article cannot be used for long
observation sequences ( $$>100$$) due to underflow problems. Recall that

$$
P(Y= Y_t, X = X_T\mid\lambda) =\prod_{t=1}^{T-1}q_{x_t, x_{t+1}}\prod_{t=1}^T b_{x_t}(y_t)
$$

which consists of a product of numbers between 0 and 1. For sufficiently
large $$T$$ this quantity is equal to zero for computers. Hence, we need to scale
the values of $$Q$$ and $$B$$ such that we do not have this numerical issue.

The standard approach is to standardise $$\alpha_t(j)$$ and $$\beta_t(j)$$ by normalising the values for each $$j\in [N]$$ by multiplying these values with

$$
c_t =\frac{1}{\sum_{i=1}^N\alpha_t(i)}
$$

to obtain

$$
\hat{\alpha}_t(j) =\alpha_t(j) c_t
$$

and

$$
\hat{\beta}_t(j) =\beta_t(j) c_t.
$$

    ForwardProbAlgScaled = function(Y, V, pi, B, Q) {
      # Define empty forward matrix
      forward <- matrix(data=0, nrow=length(V), ncol=length(Y))
      c_scale <- 1 / rep(1, ncol=length(Y))

      # Fill first elements
      forward[,1] <- B[Y[1],] * pi
      c_scale[1] <- sum(forward[,1])
      forward[,1] <- forward[,1] * c_scale[1]

      # Now for all other time steps
      for (t in 2:length(Y)) {
    tmp <- forward[,t-1] %*% Q * B[Y[t],]
    c_scale[t] <- 1 / sum(tmp)
    forward[,t] <- tmp * c_scale[t]
      }

      return(list(forward, c_scale))
    }

    BackProbAlgScaled = function(Y, V, pi, B, Q, c_scale) {
    # Define empty forward matrix
      back_prob <- matrix(data=0, nrow=length(V), ncol=length(Y))
      T_max <- length(Y)

      # Fill first elements
      back_prob[, T_max] <- rep(1, length(V)) * c_scale[T_max]

      # Now for all other time steps
      for (t in (T_max-1):1) {
    back_prob[,t] <- Q %*% (back_prob[, t+1] * B[Y[t+1], ]) * c_scale[t]
      }

      return(back_prob)
    }

    ComputeGammaScaled <- function(alpha, beta) {
      # Obtain gamma for all t and v
      alpha_beta <- alpha * beta
      gamma <- t(t(alpha_beta) / apply(alpha_beta, 2, sum))
    }

    ComputePsiScaled <- function(alpha, beta, B, Q, Y) {
      # Create empty matrix
      T_max <- ncol(beta)
      psi <- array(-1, dim=c(T_max-1, nrow(Q), nrow(Q)))

      for (t in 1:(T_max-1)) {
    psi[t,, ] <- cbind(alpha[, t], alpha[, t]) * Q * rbind(B[Y[t+1], ], B[Y[t+1], ]) * rbind(beta[, t+1], beta[, t+1])
    psi[t,, ] <- psi[t,, ] / sum(psi[t,, ])
      }

      return(psi)
    }

The adapted Balm-Welch algorithm is then given by

    BalmWelchAlg = function(Y, Q, B, pi, V, D) {
      converged = FALSE
      max_iter = 1000
      iter = 1
      prob_old = log(1e-12)

      while(!converged & iter!= max_iter) {
    # First compute alpa and beta
    forward_list <- ForwardProbAlgScaled(Y, V, pi, B, Q)
    c_scale <- forward_list[[2]]
    alpha <- forward_list[[1]]
    beta <- BackProbAlgScaled(Y, V, pi, B, Q, c_scale)
    prob_new <- log(1 / prod(c_scale))

    # E-Step
    gamma <- ComputeGammaScaled(alpha, beta)
    psi <- ComputePsiScaled(alpha, beta, B, Q, Y)

    # M-Step
    Q <- ComputeQ(psi, gamma)
    B <- ComputeBHat(gamma, Y, D)
    pi <- gamma[, 1]

    # Update stopping rules
    if (prob_new - prob_old < 1e-4) {
      converged <- TRUE
    }
    prob_old = prob_new
    iter <- iter + 1
      }

      return(list(Q = Q, B = B, pi = pi))
    }

    model <- BalmWelchAlg(Y, Q, B, pi, V, D)

    model[["Q"]]

    ##  [,1] [,2]
    ## [1,]1 4.738597e-19
    ## [2,]1 4.788997e-18

    model[["B"]]

    ##  [,1] [,2]
    ## [1,] 1.974110e-43 1.000000e+00
    ## [2,] 3.333333e-01 9.278520e-20
    ## [3,] 6.666667e-01 5.643931e-18

    model[["pi"]]

    ## [1] 5.922329e-43 1.000000e+00
