from __future__ import annotations
from functools import partial
from typing import Tuple, Callable

import jax
from jax._src.numpy.lax_numpy import ndarray
import jax.numpy as jnp
import jax.random as random
import jax.lax as lax

import numpy as np
from tqdm.auto import tqdm

# This code adapted from the original jaxnuts code (https://github.com/guillaume-plc/jaxnuts) under MIT License, reproduced below
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def force_cpu() -> None:
    """Forces JAX to run on the CPU.
    
    Should be called right after importing the module.
    """
    jax.config.update('jax_platform_name', 'cpu')

class NUTS:
    """No U-turn sampler

    A python implementation of the No U-turn sampler from [Hoffman and Gelman, 2011](https://arxiv.org/pdf/1111.4246.pdf), 
    Algorithm 6. This implementation uses JAX for efficient gradients computation (although a user-defined 
    value-and-gradient function can also be used), and to speed up execution by JIT-compiling the parts of 
    the code that can be (due to the recursive nature of the main algorithm, the main loop cannot be entirely compiled).
    
    *Disclaimer: JIT-compiling necessitates to split the code in many small functions and to use LAX primitives 
    to handle 'if' statements, which makes the code quite unreadable.*

    **NOTE:** In low dimensions, GPU overhead is not negligible and makes this code very slow. In this case,
    a very significant performance increase can be obtained by forcing JAX to run on the CPU. This can be 
    achieved either by calling `jaxnuts.sampler.force_cpu()` right after importing the module, or alternatively 
    using `jax.config.update('jax_platform_name', 'cpu')` right after importing JAX.

    Attributes
    ----------
    theta0 : ndarray
        Initial value to start sampling from.
    target_acceptance : float
        The target acceptance rate for the dual-averaging scheme used to choose the step size
        (default .6).
    M_adapt : int
        The number of steps used for the dual averaging scheme (default 1000).
    delta_max : float
        A large constant used to check that the last leapfrog step was not completely 
        off-track (default 1000.0).
    logp : Callable[[ndarray], float] | None
        A function returning the logarithm of the probability density to sample from
        (not necessarily normalized). If it is provided, its gradient will be computed
        using 'jax.grad'. Otherwise, 'logp_and_grad' is expected to be provided.
    logp_and_grad : Callable[[ndarray], Tuple[float, ndarray]] | None
        A function returning the logarithm of the probability density to sample from
        as well as its gradient. If it is provided, any function supplied to 'logp'
        will be ignored.

    Methods
    -------
    sample(M, key)
        Draw M samples of the input distribution.
    """
    def __init__(self, theta0: ndarray, logp: Callable[[ndarray], float] = None, 
                 logp_and_grad: Callable[[ndarray], Tuple[float, ndarray]] = None, 
                 target_acceptance: float = .6, M_adapt: int = 1000, delta_max: float = 1000) -> None:
        if(theta0.ndim != 1):
            raise ValueError("The initial value should be 1-dimensional vector.")
        self.theta0 = theta0 # Initial value
        self.target_acceptance = target_acceptance # Target acceptance for the dual-averaging scheme (to choose epsilon)
        self.M_adapt = max(1, M_adapt) # Number of iterations used for the dual-averaging scheme
        self.delta_max = delta_max # Some large constant used to check that the leapfrog step is not completely off-track
        # Log-probability and its gradient :
        if logp is None and logp_and_grad is None:
            raise ValueError("Please provide either the log-probability function or a function returning the log-probability and its gradient.")
        if not(logp is None):
            self.logp = logp
            self.gradp = jax.jit(jax.grad(self.logp))
            self.logp_and_grad = jax.jit(jax.value_and_grad(self.logp))
        elif not(logp_and_grad is None):
            self.logp_and_grad = logp_and_grad

    def sample(self, M: int, key: int) -> Tuple[ndarray, ndarray, float]:
        """Draw M samples from the input distribution

        Start by finding a reasonable starting value for the step size,
        then run M_adapt + M steps of the NUTS algorithm, where a dual-averaging scheme is
        used in the first M_adapt steps to automatically adpat the value of the step size.

        Parameters
        ----------
        M : int
            The number of samples to draw.
        key : ndarray
            A PRNG key, used to feed the random generators used by the algorithm.

        Returns
        -------
        key : ndarray
            The updated PRNG key.
        samples : ndarray
            An array of shape (M, D) were D is the dimension of a single sample,
            containing the generated samples.
        epsilon : float
            The value of the step size at the end of the dual-averaging scheme.
        """
        pbar = tqdm(total=self.M_adapt + M, desc="Looking for a reasonable step size")
        
        # Heuristic to find a reasonable initial step size
        key, eps = self._findReasonableEpsilon(self.theta0, key)
        
        # Initialize variables
        mu = jnp.log(10 * eps)
        eps_bar = 1.0
        H_bar = 0.0
        gamma = .05
        t_0 = 10
        kappa = .75
        
        # Record samples: use a plain numpy array because jax indexed assignement 
        # is slow (performs full array copies).
        thetas = np.zeros((self.M_adapt + M + 1, self.theta0.size))
        thetas[0] = self.theta0

        # Sampling
        pbar.set_description("Adapting step size")
        for m in range(1, self.M_adapt + M + 1):
            # Initialize momentum and pick a slice, record the initial log joint probability
            key, r, logu, logjoint0 = self._init_iteration(thetas[m-1], key)
            
            # Initialize the trajectory
            theta_m, theta_p, r_m, r_p = thetas[m-1], thetas[m-1], r, r
            j = 0 # Trajectory iteration
            n = 1 # Length of the trajectory
            s = 1 # Stop indicator
            thetas[m] = thetas[m-1] # If everything fails, the new sample is our last position
            while s == 1:
                # Choose a direction
                key, v = self._draw_direction(key)
                
                # Double the trajectory length in that direction
                if v == -1:
                    key, theta_m, r_m, _, _, theta_f, n_f, s_f, alpha, n_alpha = self._build_tree(theta_m, r_m, logu, v, j, eps, logjoint0, key)
                else:
                    key, _, _, theta_p, r_p, theta_f, n_f, s_f, alpha, n_alpha = self._build_tree(theta_p, r_p, logu, v, j, eps, logjoint0, key)

                # Update theta with probability n_f / n, to effectively sample a point from the trajectory;
                # Update the current length of the trajectory;
                # Update the stopping indicator: check it the trajectory is making a U-turn.
                key, thetas[m], n, s, j = self._trajectory_iteration_update(thetas[m], n, s, j, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, key)

            if m <= self.M_adapt:
                # Dual averaging scheme to adapt the step size 'epsilon'.
                H_bar, eps_bar, eps = self._adapt(mu, eps_bar, H_bar, t_0, kappa, gamma, alpha, n_alpha, m)
            elif m == self.M_adapt + 1:
                pbar.set_description("Sampling")
                eps = eps_bar

            pbar.update()
        pbar.close()
        return key, thetas[1 + self.M_adapt:], eps

    @partial(jax.jit, static_argnums=0)
    def _init_iteration(self, theta: ndarray, key: ndarray) -> Tuple[ndarray, ndarray, float, float]:
        """Initialize the sampling iteration

        Parameters
        ----------
        theta : ndarray
            The previous sample
        key : ndarray
            The PRNG key.

        Returns
        -------
        key : ndarray
            The updated PRNG key.
        r : ndarray
            The initial momentum vector.
        logu : float
            The slice for this iteration.
        logjoint : float
            The logarithm of the joint probability p(theta, r)
        """
        key, *subkeys = random.split(key, 3)
        r = random.normal(subkeys[0], shape=self.theta0.shape)
        logprob, _ = self.logp_and_grad(theta)
        logjoint = logprob - .5 * jnp.dot(r, r)
        logu = jnp.log(random.uniform(subkeys[1])) + (logjoint)
        return key, r, logu, logjoint

    @partial(jax.jit, static_argnums=0)
    def _draw_direction(self, key: ndarray) -> Tuple[ndarray, int]:
        """Draw a random direction (-1 or 1)"""
        key, subkey = random.split(key)
        v = 2 * random.bernoulli(subkey) - 1
        return key, v

    @partial(jax.jit, static_argnums=0)
    def _trajectory_iteration_update(self, theta: ndarray, n: int, s: int, j: int, 
                                     theta_m: ndarray, r_m: ndarray, theta_p: ndarray, r_p: ndarray, 
                                     theta_f: ndarray, n_f: int, s_f: int, key: int) -> Tuple[ndarray, ndarray, int, int, int]:
        """Update trajectory parameters

        Parameters
        ----------
        theta : ndarray
            The previous sample.
        n : int
            Previous length of the trajectory.
        s : int
            Previous stopping indicator. 
        j : int
            Trajectory iteration index
        theta_m : ndarray
            Trajectory tail.
        r_m : ndarray
            Tail momentum.
        theta_p : ndarray
            Trajectory head.
        r_p : ndarray
            Head momentum
        theta_f : ndarray
            Sample from the last trajectory sub-tree.
        n_f : int
            Size of the last trajectory sub-tree
        s_f : int
            Stopping indicator of the last trajectory sub-tree
        key : ndarray
            PRNG key.

        Returns
        -------
        key : ndarray
            Updated PRNG key.
        theta : ndarray
            Updated sample
        n : int
            Updated trajectory size.
        s : int
            Updated stopping indicator.
        j : int
            Updated iteration index.
        """
        # If we are not stopping here, update theta with probability n_f / n
        operand = key, theta, theta_f, n, n_f
        key, theta = lax.cond(s_f == 1, self._draw_theta , lambda op: op[:2], operand)
        # Update the trajectory length
        n += n_f
        # Check if we are making a U-turn
        s = s_f * (jnp.dot(theta_p - theta_m, r_m) >= 0) * (jnp.dot(theta_p - theta_m, r_p) >= 0)
        # Update iteration index
        j += 1
        return key, theta, n, s, j

    @partial(jax.jit, static_argnums=0)
    def _draw_theta(self, operand: Tuple[ndarray, ndarray, ndarray, int, int]) -> Tuple[ndarray, ndarray]:
        """Replace the last sample with the new one with probability n_f / n.
        
        Parameters
        ----------
        operand : Tuple[ndarray, ndarray, ndarray, int, int]
            A tuple containing the PRNG key, the old sample, the new one, the previous total
            length of the trajectory and the length of the new sub-tree.

        Returns
        -------
        result : Tuple[ndarray, ndarray]
            A tuple containing the updated PRNG key and the chosen sample.
        """
        key, theta, theta_f, n, n_f = operand
        key, subkey = random.split(key)
        return lax.cond(random.uniform(subkey) < lax.min(1., n_f.astype(float) / n),
               lambda op: (op[0], op[2]), lambda op: op[:2], (key, theta, theta_f, n, n_f))

    @partial(jax.jit, static_argnums=0)
    def _adapt(self, mu: float, eps_bar: float, H_bar: float, t_0: float, 
               kappa: float, gamma: float, alpha: float, n_alpha: int, m: int) -> Tuple[float, float, float]:
        """Update the step size according to the dual averaging scheme.
        
        Parameters
        ----------
        mu : float
            Value towards which the iterates (epsilon) are shrinked.
        eps_bar : float
            Averaged iterate of the dual-averaging scheme.
        eps : float
            Iterate of the dual-averaging scheme.
        H_bar : float
            Averaged difference of the current pseudo-acceptance rate to the desired one.
        t_0 : float
            Free parameter to stabilize the initial iterations.
        kappa : float
            Power of the step size schedule.
        gamma : float
            Free parameter that controls the amount of shrinkage towards mu.
        alpha : float
            Pseudo-acceptance probability of the last trajectory.
        n_alpha : float
            Size of the last trajectory.
        m : int
            Iteration index

        Returns
        -------
        H_bar : float
            Updated averaged difference of the current pseudo-acceptance rate to the desired one.
        eps_bar : float
            Updated averaged iterate.
        eps : float
            Updated iterate.
        """
        eta = 1 / (m + t_0)
        H_bar = (1 - eta) * H_bar + eta * (self.target_acceptance - alpha / n_alpha)
        mkappa = m**(-kappa)
        eps = jnp.exp(mu - (jnp.sqrt(m) / gamma) * H_bar)
        eps_bar = jnp.exp(mkappa * jnp.log(eps) + (1 - mkappa) * jnp.log(eps_bar))
        return H_bar, eps_bar, eps

    def _build_tree(self, theta: ndarray, r: ndarray, logu: float, v: int, j: int, 
                   eps: float, logjoint0: ndarray, key: ndarray) -> Tuple[ndarray, 
                   ndarray, ndarray, ndarray, ndarray, ndarray, int, int, float, int]:
        """Recursively build the trajectory binary tree.
        
        Parameters
        ----------
        theta : ndarray
            Sample position.
        r : ndarray
            Sample momentum.
        logu : float
            Slice position.
        v : int
            Direction to take.
        j : int
            Iteration index of the trajectory.
        eps : float
            Step size.
        logjoint0 : ndarray
            Logarithm of the joint probability p(theta, r) of the
            original sample.
        key : ndarray
            PRNG key.
        
        Returns
        -------
        key : ndarray
            Updated PRNG key
        theta_m : ndarray
            Tail position
        r_m : ndarray
            Tail momentum
        theta_p : ndarray
            Head position
        r_p : ndarray
            Head momentum
        theta_f : ndarray
            Sampled position
        n_f : int
            Slice set size.
        s_f : int
            Stop indicator.
        alpha_f : float
            Pseudo acceptance rate.
        n_alpha_f : int
            Total set size.
        """
        if j == 0: # Initialize the tree
            return self._init_build_tree(theta, r, logu, v, j, eps, logjoint0, key)
        else: # Recurse
            key, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, alpha_f, n_alpha_f = self._build_tree(theta, r, logu, v, j - 1, eps, logjoint0, key)
            if s_f == 1: # If no early stopping, recurse.
                if v == -1:
                    key, theta_m, r_m, _, _, theta_ff, n_ff, s_ff, alpha_ff, n_alpha_ff = self._build_tree(theta_m, r_m, logu, v, j - 1, eps, logjoint0, key)
                else:
                    key, _, _, theta_p, r_p, theta_ff, n_ff, s_ff, alpha_ff, n_alpha_ff = self._build_tree(theta_p, r_p, logu, v, j - 1, eps, logjoint0, key)
                
                # Update theta with probability n_ff / (n_f + n_ff);
                # Update the stopping indicator (U-turn);
                # Update the pseudo-acceptance;
                # Update the length counters;
                key, theta_f, n_f, s_f, alpha_f, n_alpha_f = self._update_build_tree(theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, alpha_f, n_alpha_f, theta_ff, n_ff, s_ff, alpha_ff, n_alpha_ff, key)
            return key, theta_m, r_m, theta_p, r_p, theta_f, n_f, s_f, alpha_f, n_alpha_f
        
    @partial(jax.jit, static_argnums=0)
    def _init_build_tree(self, theta : ndarray, r : ndarray, logu : float, v : int, j : int, 
                         eps : float, logjoint0 : float, key : ndarray) -> Tuple[ndarray, 
                         ndarray, ndarray, ndarray, ndarray, ndarray, int, int, float, int]:
        """Initialize the trajectory binary tree."""
        # Perform one leapfrog step
        theta, r, logp, _ = self._leapfrog(theta, r, v * eps)
        logjoint = logp - .5 * jnp.dot(r, r)
        # Check if the step is within the slice
        n_f = (logu <= logjoint).astype(int)
        # Check that we are not completely off-track
        s_f = (logu < logjoint + self.delta_max).astype(int)
        # Compute the acceptance rate
        prob_ratio = jnp.exp(logjoint - logjoint0)
        alpha_f = lax.cond(jnp.isnan(prob_ratio), lambda _: 0., lambda _: lax.min(prob_ratio, 1.), None) # Presumably if the probability ratio diverges,
                                                                                                         # it is because the log-joint probability diverges, i.e. the probability tends
                                                                                                         # to zero, so its log is -infinity. Then the acceptance rate of this step,
                                                                                                         # which is what 'alpha_f' stands for, should be zero.
        # Total set size
        n_alpha_f = 1
        return key, theta, r, theta, r, theta, n_f, s_f, alpha_f, n_alpha_f

    @partial(jax.jit, static_argnums=0)
    def _update_build_tree(self, theta_m: ndarray, r_m: ndarray, theta_p: ndarray, r_p: ndarray, 
                           theta_f: ndarray, n_f: int, s_f: int, alpha_f: float, n_alpha_f: int, 
                           theta_ff: ndarray, n_ff: int, s_ff: int, alpha_ff: float, n_alpha_ff: int, 
                           key: ndarray) -> Tuple[ndarray, ndarray, int, int, float, int]:   
        """Updates the tree parameters.
        
        Parameters
        ----------
        theta_m : ndarray
            Tail position
        r_m : ndarray
            Tail momentum
        theta_p : ndarray
            Head position
        r_p : ndarray
            Head momentum
        theta_f : ndarray
            First sampled position
        n_f : int
            First slice set size.
        s_f : int
            First stop indicator.
        alpha_f : float
            First pseudo acceptance rate.
        n_alpha_f : int
            First total set size.
        theta_ff : ndarray
            Second sampled position
        n_ff : int
            Second slice set size.
        s_ff : int
            Second stop indicator.
        alpha_ff : float
            Second pseudo acceptance rate.
        n_alpha_ff : int
            Second total set size.

        Returns
        -------
        key : ndarray
            Updated PRNG key
        theta_f : ndarray
            Sampled position
        n_f : int
            Slice set size.
        s_f : int
            Stop indicator.
        alpha_f : float
            Pseudo acceptance rate.
        n_alpha_f : int
            Total set size.
        """
        key, subkey = random.split(key)
        update = random.uniform(subkey)
        theta_f = lax.cond(update <= n_ff / lax.max(n_f + n_ff, 1), lambda _: theta_ff, lambda _: theta_f, None)
        alpha_f += alpha_ff
        n_alpha_f += n_alpha_ff
        s_f = s_ff * (jnp.dot(theta_p - theta_m, r_m) >= 0).astype(int) * (jnp.dot(theta_p - theta_m, r_p) >= 0).astype(int)
        n_f += n_ff
        return key, theta_f, n_f, s_f, alpha_f, n_alpha_f

    def _findReasonableEpsilon(self, theta: ndarray, key: ndarray) -> Tuple[ndarray, float]:
        """Heuristic to find a reasonable initial value for the step size.
        
        Finds a reasonable value for the step size by scaling it until the acceptance probability
        crosses 0.5 .

        Parameters
        ----------
        theta : ndarray
            The initial sample position.
        key : ndarray
            PRNG key

        Returns
        -------
        key : ndarray
            The updated PRNG key
        eps : float
            A reasonable initial value for the step size
        """
        eps = 1
        key, subkey = random.split(key)
        r = random.normal(subkey, shape=theta.shape)

        logp, gradlogp = self.logp_and_grad(theta)
        if jnp.isnan(logp):
            raise ValueError("log probability of initial value is NaN.")

        theta_f, r_f, logp, gradlogp = self._leapfrog(theta, r, eps)

        # First make sure that the step is not too large i.e. that we do not get diverging values.
        while jnp.isnan(logp) or jnp.any(jnp.isnan(gradlogp)):
            eps /= 2
            theta_f, r_f, logp, gradlogp = self._leapfrog(theta, r, eps)
        
        # Then decide in which direction to move
        logp0, _ = self.logp_and_grad(theta)
        logjoint0 = logp0 - .5 * jnp.dot(r, r)
        logjoint = logp - .5 * jnp.dot(r_f, r_f)
        a = 2 * (logjoint - logjoint0 > jnp.log(.5)) - 1
        # And successively scale epsilon until the acceptance rate crosses .5
        while a * (logp - .5 * jnp.dot(r_f, r_f) - logjoint0) > a * jnp.log(.5):
            eps *= 2.**a
            theta_f, r_f, logp, _ = self._leapfrog(theta, r, eps)
        return key, eps

    @partial(jax.jit, static_argnums=0)
    def _leapfrog(self, theta: ndarray, r: ndarray, eps: float) -> Tuple[ndarray, ndarray, float, ndarray]:
        """Perform a leapfrog step.
        
        Parameters
        ----------
        theta : ndarray
            Initial sample position.
        r : ndarray
            Initial momentum.
        eps : float
            Step size;

        Returns
        -------
        theta : ndarray
            New sample position
        r : ndarray
            New momentum
        logp : float
            Log probability of the new position.
        gradlogp : ndarray
            Gradient of the log probability evaluated
            at the new position.
        """
        logp, gradlogp = self.logp_and_grad(theta)
        r = r + .5 * eps * gradlogp
        theta = theta + eps * r
        logp, gradlogp = self.logp_and_grad(theta)
        r = r + .5 * eps * gradlogp
        return theta, r, logp, gradlogp


