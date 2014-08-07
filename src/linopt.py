"""
Module ltiopt.py: Dynamic optimization of LTI systems
"""
# Time-stamp: <Last change 2013-06-20 09:58:29 by Steffen Waldherr>

from __future__ import division

import operator
from copy import copy
import warnings
from itertools import product

import numpy as np
from numpy.polynomial.legendre import legroots, legder, legval
from scipy import sparse
import scipy.linalg as scila
from scipy import integrate
from scipy import optimize

class Store(object):
    pass

globals = Store()

try:
    import cvxopt
    import cvxopt.solvers
    globals.have_cvxopt = True
except ImportError as err:
    globals.have_cvxopt = False
    globals.cvxopt_err = str(err)

def unitvector(n, i):
    uv = np.zeros(n)
    uv[i] = 1.0
    return uv

def block_diagonal(arrays):
    res = np.zeros((sum(a.shape[0] for a in arrays), sum(a.shape[1] for a in arrays)))
    for i,a in enumerate(arrays):
        ind0 = sum(ai.shape[0] for ai in arrays[:i])
        ind1 = sum(ai.shape[1] for ai in arrays[:i])
        res[ind0:ind0+a.shape[0], ind1:ind1+a.shape[1]] = a
    return res
    
def cvx_spdiag(x):
    """
    Sparse diagonal of not necessarily square matrices.
    """
    zsparse = lambda size: cvxopt.spmatrix([], [], [], size)
    xl = map(cvxopt.sparse, x)
    return cvxopt.sparse([[cvxopt.sparse([zsparse((sum(xj.size[0] for xj in xl[:i]), xi.size[1])),
                                          xi,
                                          zsparse((sum(xj.size[0] for xj in xl[(i+1):]), xi.size[1]))])]
                          for i,xi in enumerate(xl)])

def cvx_sparse(M):
    """
    Create and return a ``cvxopt.spmatrix`` sparse matrix from a scipy sparse matrix ``M``.
    """
    I,J = M.nonzero()
    if len(I)>0:
        D = np.array(M[I,J].todense()).flatten()
    else:
        D = []
    return cvxopt.spmatrix(D, map(int, I), map(int, J), M.shape)

class OptimalityError(Exception):
    pass

def bisection(testfun, start, task='maximize', tolerance=1e-3):
    """
    ``testfun`` is a function of a scalar number returning a tuple of a Boolean value
    and another object ``O``.
    
    If ``task`` is ``'maximize'``, find the largest ``x`` such that ``testfun(x)``
    returns ``True`` as first return value, using bisection.
    This assumes that ``testfun(x)`` returns ``True`` for ``x`` between 0 and
    a sufficiently small value.
    If ``task`` is ``'minimize'``, find the smallest such ``x``. This assumes that
    ``testfun(x)`` returns ``True`` for ``x`` above a sufficiently large value.

    Use ``start`` as an initial guess for ``x``.

    Returns the best estimate for ``x``, and the object ``O`` returned by ``testfun`` for
    this value of ``x``.

    This implementation assumes that ``testfun`` is only defined for non-negative arguments.
    """
    minx = 0.0
    maxx = np.inf
    nextx = start
    bestres = None
    while maxx - minx > tolerance:
        feasible, res = testfun(nextx)
        if task == 'maximize':
            if feasible:
                bestres = res
                minx = nextx
            else:
                maxx = nextx
            if not np.isfinite(maxx):
                nextx = 2 * minx
        else: # minimize
            if feasible:
                bestres = res
                maxx = nextx
            else:
                minx = nextx
            if not np.isfinite(maxx):
                nextx = 2 * minx
        if np.isfinite(maxx):
            nextx = 0.5 * (maxx + minx)
    if task == 'maximize':
        return minx, bestres
    else:
        return maxx, bestres
            

class Collocation(object):
    """
    Provides auxiliary routines for collocation.
    """
    basisfun_available = ("lagrange")
    collocation_available = ("lobatto","radau","radau-rev")
    def __init__(self, K, basisfun="lagrange", collocation="lobatto"):
        """
        Define a Collocation object with K > 0 collocation points.
        Implemented basis functions:
        lagrange - Lagrange polynomials
        Implemented types of collocation points:
        lobatto - Lobatto collocation points
        radau - Radau collocation points
        radau-rev - Radau collocation points with endpoint included
        """
        self.K = K
        self.basisfun = basisfun
        self.collocation = collocation
        if collocation not in self.collocation_available:
            raise ValueError("Collocation points %s not implemented, must be one of: %s" % (collocation, self.collocation_available))
        if basisfun not in self.basisfun_available:
            raise ValueError("Basis function %s not implemented, must be one of: %s" % (basisfun, self.basisfun_available))
        if K<2 and collocation == "lobatto":
            raise ValueError("Lobatto collocation requires K > 1, got K = %d." % K)

    def points(self):
        """
        Return collocation points in the interval [-1, 1].
        """
        if self.collocation == "lobatto":
            if self.K <= 2:
                return np.array([-1.0, 1.0])
            else:
                return np.concatenate(([-1.0], legroots(legder([0 for i in range(self.K-1)] + [1])), [1.0]))
        if self.collocation == "radau":
            return legroots([0 for i in range(self.K-1)] + [1, 1])
        if self.collocation == "radau-rev":
            return -legroots([0 for i in range(self.K-1)] + [1, 1])[::-1]

    def weights(self):
        """
        Return Gauss quadrature weights.
        """
        if self.collocation == "lobatto":
            if self.K > 2:
                innerweights = 2.0 / (self.K * (self.K-1) * legval(legroots(legder([0 for i in range(self.K-1)] + [1])),
                                                                   [0 for i in range(self.K-1)] + [1])**2)
            else:
                innerweights = []
            w0 = 2.0/(self.K * (self.K-1))
            return np.concatenate(([w0], innerweights, [w0]))
        if self.collocation == "radau":
            if self.K > 1:
                p = self.points()[1:]
                innerweights = 1.0 / ((1-p) * legval(p, legder([0 for i in range(self.K-1)] + [1]))**2)
            else:
                innerweights = []
            return np.concatenate(([2.0 / self.K**2], innerweights))
        if self.collocation == "radau-rev":
            if self.K > 1:
                p = -self.points()[-2::-1]
                innerweights = 1.0 / ((1-p) * legval(p, legder([0 for i in range(self.K-1)] + [1]))**2)
            else:
                innerweights = []
            return np.concatenate(([2.0 / self.K**2], innerweights))[::-1]

    def vander(self, dim=1):
        """
        Return Vandermonde matrix for vectors of length dim.
        """
        I = np.eye(dim)
        p = self.points()
        if self.basisfun == "lagrange":
            P = lambda i,j: 1.0 if i==j else 0.0
            return np.vstack(tuple(np.hstack(tuple(P(i,j) * I for i in range(self.K)))
                                             for j in range(self.K)))

    def int_basis(self, p, q):
        """
        Compute int_-1^(r_q) L_p(s) ds.
        """
        ps = self.points()
        if self.basisfun == "lagrange":
            return integrate.fixed_quad(Lp, -1, ps[q], n=int(self.K/2)+1)[0] if q>=0 else integrate.fixed_quad(Lp, -1, 1, n=int(self.K/2)+1)[0]

    def int_vander(self, dim=1):
        """
        Return integral Vandermonde matrix for vectors of length dim up to q-th collocation point
        """
        I = np.eye(dim)
        ps = self.points()
        if self.basisfun == "lagrange":
            L = lambda s,p: reduce(operator.mul, [(s - pj)/(ps[p] - pj) for pj in ps if not pj == ps[p]], 1.0)
            Pint = lambda q,p: integrate.fixed_quad(lambda s: L(s,p), -1, ps[q], n=int(self.K/2) + 1)[0]
            return np.vstack(tuple(np.hstack(tuple(Pint(q,p) * I for p in range(self.K)))
                                   for q in range(self.K)))

    def eval(self, p, X):
        """
        Evaluate collocated function at point ``p`` in the interval ``[-1, 1]``, using
        elements of the list ``X`` as basis function coefficients.
        """
        ps = self.points()
        if self.basisfun == "lagrange":
            L = lambda s,i: reduce(operator.mul, [(s - pj)/(ps[i] - pj) for j,pj in enumerate(ps) if not j == i], 1.0)
            return sum(Xi*L(p,i) for i,Xi in enumerate(X))

    def integrate(self, q, t):
        """
        Return the integral of the ``q``-th basis function over the interval ``[-1, t]``
        """
        ps = self.points()
        if self.basisfun == "lagrange":
            L = lambda s,i: reduce(operator.mul, [(s - pj)/(ps[i] - pj) for j,pj in enumerate(ps) if not j == i], 1.0)
            return integrate.fixed_quad(lambda s: L(s,q), -1, t, n=(self.K // 2 + 1))[0]

LLCollocation = lambda K: Collocation(K, "lagrange", "lobatto")
LRCollocation = lambda K: Collocation(K, "lagrange", "radau")
LRrevCollocation = lambda K: Collocation(K, "lagrange", "radau-rev")

class CollocatedTrajectory(object):
    """
    Represents a trajectory for an LTI system approximated by collocation.
    """
    def __init__(self, Z, *args): # args: collocation, finaltime, timesteps, controldim, statedim, initialstate
        if len(args):
            collocation, finaltime, timesteps, controldim, statedim, initialstate = args
            self.collocation = collocation
            self._u = Z[:collocation.K*timesteps*controldim]
            self._xdot = Z[collocation.K*timesteps*controldim:collocation.K*timesteps*(controldim+statedim)]
            self._x = Z[collocation.K*timesteps*(controldim+statedim):]
            self._m = controldim
            self._n = statedim
            self._N = timesteps
            self._tf = finaltime
            self._x0 = initialstate
            self._h = finaltime / timesteps
            self.finaltime = finaltime
        else:
            self.collocation = Z.collocation
            self._u = Z._u
            self._xdot = Z._xdot
            self._x = Z._x
            self._m = Z._m
            self._n = Z._n
            self._N = Z._N
            self._tf = Z._tf
            self._x0 = Z._x0
            self._h = Z._h
            self.finaltime = Z.finaltime
            
    def get_control(self, t):
        def get_single(ti):
            ind = self.index(ti)
            if ind >= self._N:
                ind = self._N - 1
            us = [self._u[ind*self.collocation.K*self._m+i*self._m:ind*self.collocation.K*self._m+(i+1)*self._m] for i in range(self.collocation.K)]
            return self.collocation.eval(self.mapt(ti), us)
        return get_single(t) if np.isscalar(t) else np.array([get_single(ti) for ti in t])

    def get_state(self, t):
        def get_single(ti):
            ind = self.index(ti)
            xs = self._x[(ind-1)*self._n:ind*self._n] if ind > 0 else self._x0
            i0 = ind*self.collocation.K*self._n
            if i0 < len(self._xdot):
                return xs + sum(self._h / 2 * self._xdot[i0+i*self._n:i0+(i+1)*self._n] * self.collocation.integrate(i, self.mapt(ti))
                                for i in range(self.collocation.K))
            else:
                return xs
        return get_single(t) if np.isscalar(t) else np.array([get_single(ti) for ti in t])

    def get_statederivative(self, t):
        def get_single(ti):
            ind = self.index(ti)
            if ind >= self._N:
                ind = self._N - 1
            xdots = [self._xdot[ind*self.collocation.K*self._n+i*self._n:ind*self.collocation.K*self._n+(i+1)*self._n] for i in range(self.collocation.K)]
            return self.collocation.eval(self.mapt(ti), xdots)
        return get_single(t) if np.isscalar(t) else np.array([get_single(ti) for ti in t])

    def index(self, t):
        return t // self._h

    def mapt(self, t):
        """
        Map the timepoint ``t`` to the range ``[-1, 1]`` in the respective discretization interval.
        """
        return 2 * (t / self._h - self.index(t)) - 1

class LinOpt(object):
    """
    Dynamic optimizer for an LTI system.

    Given an LTI system

        dx/dt = A x + B u + u0        (LTI)

    solves the linear optimal control problem

        min_u int_0^T (C x(t) + D u(t) + E) dt + F x(T)
        s.t.  (LTI)
              Gx x(t) + Gu u(t) + Gk <= 0
              Hx x(T) + Hk <= 0.
    """
    def __init__(self, *args):
        """
        Set up the optimizer.

        Usage:
        LinOpt(A,B): initialize the system with matrices A and B
        LinOpt(A,B,x0): initialize the system with matrices A and B, and initial condition x0
        LinOpt(A,B,x0,u0): initialize the system with matrices A and B, initial condition x0, and constant input u0
        x0 - initial condition (zero if left out)
        u0 - constant input (zero if left out)
        """
        if len(args) == 2:
            self.A, self.B = args
            self.x0 = np.zeros(self.A.shape[1])
            self.u0 = np.zeros(self.A.shape[1])
        elif len(args) == 3:
            self.A, self.B, self.x0 = args
            self.u0 = np.zeros(self.A.shape[1])
        elif len(args) == 4:
            self.A, self.B, self.x0, self.u0 = args
        else:
            raise TypeError("LinOpt expected 2 to 4 arguments, got %d." % len(args))
        self.m = self.B.shape[1]
        self.n = self.A.shape[1]
        self.solver = None
        self.path_constraints = []
        self.terminal_constraints = []
        self.params = {"timesteps": 50,
                       "finaltime": 1.0,
                       "fixedterm": True,
                       "collocation": Collocation(3, 'lagrange', 'radau'),
                       "finaltimetolerance": 0.01
                       }
        self.solverparams = {}
        self.set_objective(np.zeros(self.n), np.zeros(self.m), 0, np.zeros(self.n))

    def set_objective(self, C, D, E, F):
        """
        Set the objective functional for this optimization problem.

        The objective functional is given by J = int_0^T (C' x(t) + D' u(t) + E) dt + F' x(T).

        Arguments:
        C, F - vectors of the same size as the state variable x.
        D - a vector of the same size as the control variable u.
        E - a real number.

        Any of the arguments can be a scalar 0 or None to disregard the corresponding part in the objective.
        """
        self.C, self.D, self.E, self.F = C, D, E, F

    def add_path_constraint(self, Gxi, Gui, Gki):
        """
        Add a path constraint to the optimization problem.

        The path constraint is given by:
               Gxi x(t) + Gui u(t) + Gki <= 0.
        Arguments:
        Gxi - an k x n matrix, or scalar for all equal values.
        Gui - a k x m matrix, or scalar for all equal values.
        Gki - a vector of length k.
        """
        k = 1 if np.isscalar(Gki) else len(Gki)
        if np.isscalar(Gxi):
            Gxi = Gxi * np.ones((k,self.n))
        if np.isscalar(Gui):
            Gui = Gui * np.ones((k,self.m))
        if np.isscalar(Gki):
            Gki = Gki * np.ones(k)
        self.path_constraints.append((Gxi, Gui, Gki))

    def add_terminal_constraint(self, Hxi, Hki):
        """
        Add a terminal constraint to the optimization problem.

        The terminal constraint is given by:
               Hxi x(t) + Hki <= 0.
        Arguments:
        Hxi - an k x n matrix, or scalar for all equal values.
        Hki - a vector of length k.
        """
        k = 1 if np.isscalar(Hki) else len(Hki)
        if np.isscalar(Hxi):
            Hxi = Hxi * np.ones((k, self.n))
        if np.isscalar(Hki):
            Hki = Hki * np.ones(k)
        self.terminal_constraints.append((Hxi, Hki))

    def set_solver(self, solver):
        """
        Choose numerical solver to perform the optimization.

        Available solvers:
        cvxopt - Interior point solver cvxopt, requires the cvxopt module.
        """
        self.solver = solver
        if solver == "cvxopt":
            import cvxopt
            import cvxopt.solvers
            self.cvxopt = cvxopt
        else:
            raise ValueError("Unknown solver: %s" % solver)
        
    def set_parameters(self, **kwargs):
        """
        Set optimization parameters.

        Available parameters are:

        timesteps - number of discretization time steps (default = 50)
        finaltime - value of T (default = 1), maximal T in the problem with free final time
        finaltimetolerance - termination tolerance on the final time (default = 0.01)
        fixedterm - whether to keep final time fixed or not (default = True)
        collocation - which collocation scheme to use (default = Collocation(3, 'lagrange', 'radau'))

        Depending on the solver, additional parameters may be available.
        cvxopt: Parameters 'solver', 'primalstart', 'dualstart' from cvxopt.solvers.lp
        """
        for k in self.params:
            if k in kwargs:
                self.params[k] = kwargs.pop(k)
        self.solverparams.update(kwargs)

    def discretize(self, collocation=None):
        """
        Perform the discretization with collocation (use from parameters by default).
        The control, state derivative, and state trajectories are discretized as
        u(t_i + h/2 (r_j + 1)) = P ui (ui is a vector of length mK and P depends on collocation type)
        xdot(t_i + h/2 (r_j + 1)) = P xdoti (xdoti is a vector of length nK)
        xi(t_i) = x_i
        where t_i are the time interval boundaries, r_j the relative positions of the collocation points within each interval,
        and h the length of the time intervals.

        Optimization is then done over the vector z = (u11, ..., u1K, ..., uN1, ..., uNK,
                                                       xdot11, ..., xdot1K, ..., xdotN1, ..., xdotNk,
                                                       x1, ..., xN)

        Returns vectors c,b,d, a scalar e, and sparse matrices A,C corresponding to the linear program
            min  c' z + e
            s.t. A z = b
                 C z <= d
        which is the discrete approximation to the linear optimal control problem.
        """
        if collocation is None:
            collocation = self.params['collocation']
        optimdim = self.params["timesteps"] * (self.m * collocation.K + self.n * collocation.K + self.n)
        weights = collocation.weights()
        cpoints = collocation.points()
        h = self.params["finaltime"] / self.params["timesteps"]
        N = self.params["timesteps"]
        Pu = collocation.vander(self.m)
        Px = collocation.vander(self.n)
        Qu = collocation.int_vander(self.m)
        Qx = collocation.int_vander(self.n)
        xdotoffset = self.params["timesteps"] * self.m * collocation.K
        xoffset = xdotoffset + self.params["timesteps"] * self.n * collocation.K
        In = np.eye(self.n)
        Im = np.eye(self.m)
        Wn = np.hstack(tuple(w * In for w in weights))
        Wm = np.hstack(tuple(w * Im for w in weights))
        W = np.atleast_2d(weights)
        
        # discrete objective 
        if callable(self.D):
            objtup = tuple(h / 2 * W.dot(block_diagonal([np.atleast_2d(self.D(i * h + h / 2 + h / 2 * r)) for r in cpoints])).dot(Pu).ravel()
                           for i in range(self.params['timesteps']))
        elif self.D is not None and np.any(self.D):
            DWP = h / 2 * self.D.dot(Wm).dot(Pu)
            objtup = self.params["timesteps"] * (DWP.ravel(),)
        else:
            objtup = self.params["timesteps"] * (np.zeros(self.m * collocation.K),)
        if callable(self.C):
            objtup = objtup + tuple(h**2 / 4 * W.dot(block_diagonal([np.atleast_2d(self.C(i * h + h / 2 + h / 2 * r)) for r in cpoints])).dot(Qx).ravel()
                                    for i in range(self.params['timesteps']))
            objtup = objtup + tuple(h * self.C(i * h) for i in range(self.params['timesteps'] - 1))
            objoffset = h * self.C(0.0).dot(self.x0)
        elif self.C is not None and np.any(self.C):
            CWQ = h**2 / 4 * self.C.dot(Wn).dot(Qx)
            objtup = objtup + self.params["timesteps"] * (CWQ.ravel(),)
            objtup = objtup + (self.params["timesteps"]-1) * (h * self.C,)
            objoffset = h * self.C.dot(self.x0)
        else:
            objtup = objtup + self.params["timesteps"] * (np.zeros(self.n * collocation.K),)
            objtup = objtup + (self.params["timesteps"]-1) * (np.zeros(self.n),)
            objoffset = 0.0
        if self.F is not None and np.any(self.F):
            objtup = objtup + (self.F,)
        else:
            objtup = objtup + (np.zeros(self.n),)
        objvec = np.concatenate(objtup)
        assert len(objvec) == optimdim
        if callable(self.E):
            objoffset += quad(self.E, 0, self.params["finaltime"])[0]
        elif self.E is not None and self.E != 0:
            objoffset += self.E * self.params["finaltime"]

        # equality constraints
        eqmatrix = sparse.lil_matrix((self.params["timesteps"] * (collocation.K * self.n + self.n), optimdim))
        nK = self.n * collocation.K
        mK = self.m * collocation.K
        AQ = block_diagonal(collocation.K * (self.A,)).dot(Qx)
        BP = block_diagonal(collocation.K * (self.B,)).dot(Pu)
        for i in range(self.params["timesteps"]):
            # discrete dynamics (collocation)
            eqmatrix[i*nK:(i+1)*nK, i*mK:(i+1)*mK] = - BP
            eqmatrix[i*nK:(i+1)*nK, xdotoffset+i*nK:xdotoffset+(i+1)*nK] = Px - AQ * h / 2
            if i>0:
                eqmatrix[i*nK:(i+1)*nK, xoffset+(i-1)*self.n:xoffset+i*self.n] = np.vstack(collocation.K * (- self.A,))
            # continuity constraints
            if i>0:
                eqmatrix[N*nK+i*self.n:N*nK+(i+1)*self.n, xoffset+(i-1)*self.n:xoffset+i*self.n] = In
            eqmatrix[N*nK+i*self.n:N*nK+(i+1)*self.n, xdotoffset+i*nK:xdotoffset+(i+1)*nK] = Wn * h / 2
            eqmatrix[N*nK+i*self.n:N*nK+(i+1)*self.n, xoffset+i*self.n:xoffset+(i+1)*self.n] = - In
        eqvector = np.concatenate(collocation.K * self.params["timesteps"] * (self.u0,) + (-self.x0,) + 
                                  (self.params["timesteps"] - 1) * (np.zeros(self.n),))
        for i in range(collocation.K):
            eqvector[i*self.n:(i+1)*self.n] += self.A.dot(self.x0)

        # inequality constraints
        if len(self.path_constraints):
            Gx = np.vstack(tuple(Gxi for (Gxi, Gui, Gki) in self.path_constraints))
            Gu = np.vstack(tuple(Gui for (Gxi, Gui, Gki) in self.path_constraints))
            Gk = np.concatenate(tuple(Gki for (Gxi, Gui, Gki) in self.path_constraints))
            numpath = Gx.shape[0] * collocation.K * N
            GuP = block_diagonal(collocation.K * (Gu,)).dot(Pu)
            GxQ = block_diagonal(collocation.K * (Gx,)).dot(Qx)
            nG = Gx.shape[0]
        else:
            numpath = 0
        if len(self.terminal_constraints):
            Hx = np.vstack(tuple(Hxi for (Hxi, Hki) in self.terminal_constraints))
            Hk = np.concatenate(tuple(Hki for (Hxi, Hki) in self.terminal_constraints))
            numterm = Hx.shape[0]
            nH = Hx.shape[0]
        else:
            numterm = 0
        if numpath + numterm > 0:
            ineqmatrix = sparse.lil_matrix((numpath + numterm, optimdim))
            for i in range(self.params['timesteps']):
                if numpath:
                    ineqmatrix[i*collocation.K*nG:(i+1)*collocation.K*nG, i*mK:(i+1)*mK] = GuP
                    ineqmatrix[i*collocation.K*nG:(i+1)*collocation.K*nG,
                               xdotoffset+i*nK:xdotoffset+(i+1)*nK] = GxQ * h / 2
                    if i > 0:
                        ineqmatrix[i*collocation.K*nG:(i+1)*collocation.K*nG,
                                   xoffset+(i-1)*self.n:xoffset+i*self.n] = np.vstack(collocation.K * (Gx,))
            if numterm:
                ineqmatrix[N*collocation.K*nG:, xoffset+(N-1)*self.n:xoffset+N*self.n] = Hx
        else:
            ineqmatrix = np.zeros((0,optimdim))
            
        ineqvector = np.zeros(numpath + numterm)
        if numpath:
            ineqvector[:numpath] = - np.concatenate(collocation.K * N * (Gk,))
            ineqvector[:collocation.K*nG] -= np.concatenate(collocation.K * (Gx.dot(self.x0),))
        if numterm:
            ineqvector[numpath:] = - Hk
            
        return objvec, eqvector, ineqvector, objoffset, eqmatrix, ineqmatrix

    def solve(self, solver=None, params={}):
        """
        Solve the linear dynamic optimization problem.

        See LtiOpt.set_solver and LtiOpt.set_parameters for details on method arguments.

        For problems with fixed terminal time, return the optimal value and a
        :py:class:`.CollocatedTrajectory` object representing an optimal trajectory.
        For problems with variable terminal time, return a tuple of the optimal time,
        the objective value, and the trajectory.
        """
        if self.params['fixedterm']:
            return self.solve_fixedterm(solver, params)
        else:
            # First find the "best" time where the problem is still feasible.
            other = copy(self)
            other.set_objective(0*self.C, 0*self.D, 0*self.E, 0*self.F)
            other.params = self.params.copy()
            def testfun(t):
                other.set_parameters(finaltime=t)
                try:
                    res = other.solve_fixedterm(solver, params)
                    return True, res
                except OptimalityError:
                    return False, None
            topt, resopt = bisection(testfun, self.params['finaltime'], task='minimize' if self.E >= 0 else 'maximize',
                                     tolerance=self.params['finaltimetolerance'])
            # With a state- and control-independent objective, we're done.
            if np.allclose(self.C, 0) and np.allclose(self.D, 0) and np.allclose(self.F, 0):
                return (topt,) + resopt
            # Otherwise, do a scalar optimization between the "best" time and the user-supplied final time
            other = copy(self)
            other.params = self.params.copy()
            def objfun(t):
                other.set_parameters(finaltime=t)
                try:
                    res = other.solve_fixedterm(solver, params)[0]
                except OptimalityError:
                    res = np.inf
                return res
            mint = min(topt, self.params['finaltime'])
            maxt = max(topt, self.params['finaltime'])
            topt = optimize.brent(objfun, brack=(mint, maxt))
            other.set_parameters(finaltime=topt)
            return (topt,) + other.solve_fixedterm(solver, params)

    def solve_fixedterm(self, solver='cvxopt', params={}):
        """
        Solve the LTI optimization problem with cvxopt with fixed final time.

        If the optimization is successful, the method returns:

        * The optimal value.
        * A :py:class:`.CollocatedTrajectory` object representing an optimal trajectory.

        See `self.solve_lp` for available solvers and their options.
        """
        c,b,d,e,Me,Mi = self.discretize(collocation=self.params['collocation'])
        x, obj = self.solve_lp(c, b, d, Me, Mi, solver=solver, params=params)
        return (obj + e,
                CollocatedTrajectory(x, self.params['collocation'], self.params['finaltime'], self.params['timesteps'],
                                     self.m, self.n, self.x0)
            )
    
    def control_variability(self, controlvec, times, optval=None, **kwargs):
        """
        Solve the state variability problem at given time points.

        Arguments::
        controlvec - Get variability in controlvec' * u(t)
        times - list of time points at which control variability should be computed.
        optval - Objective functional value which should be achieved. If None, solve
            the optimization problem first.
        Additional arguments are passed to the ``self.optimize`` method.
        """
        collocation = self.params['collocation']
        cpoints = collocation.points()
        optimdim = self.params["timesteps"] * (self.m * collocation.K + self.n * collocation.K + self.n)
        h = self.params["finaltime"] / self.params["timesteps"]
        N = self.params["timesteps"]
        xdotoffset = self.params["timesteps"] * self.m * collocation.K
        xoffset = xdotoffset + self.params["timesteps"] * self.n * collocation.K
        
        c,b,d,e,Me,Mi = self.discretize(collocation=self.params['collocation'])
        if optval is None:
            _, optval = self.solve_lp(c, b, d, Me, Mi, params=kwargs)
        else:
            optval = optval - e
        Miext = sparse.vstack((Mi, c)).tolil()
        dext = np.concatenate((d, np.atleast_1d(optval)))
        times = np.atleast_1d(times)
        controlvec = np.atleast_1d(controlvec)
        control_min = np.zeros(len(times))
        control_max = np.zeros(len(times))
        # find indices of collocation points which are required for interpolation
        collpoints = []
        for i,ti in enumerate(times):
            ind = min(int(np.floor(ti / h)), N-1)
            rq = 2 * (ti - ind * h) / h - 1
            if rq >= cpoints[0]:
                collpoints.append((ind, cpoints[cpoints<=rq][-1]))
            elif ind > 0:
                collpoints.append((ind - 1, cpoints[-1]))
            else:
                collpoints.append((0, cpoints[0]))
            if rq <= cpoints[-1]:
                collpoints.append((ind, cpoints[cpoints>=rq][0]))
            elif ind < N - 1:
                collpoints.append((ind + 1, cpoints[0]))
            else:
                collpoints.append((N - 1, cpoints[-1]))
        collpoints = set(collpoints)
        uminmax = {}
        for ind, rq in collpoints:
            ti = ind * h + 0.5 * (rq + 1) * h
            ci = np.concatenate(ind * (np.zeros(collocation.K * self.m),) +              # u coefficients before this time interval
                                tuple(controlvec * collocation.eval(rq, unitvector(collocation.K, j)) for j in range(collocation.K)) +
                                (N-ind-1)* (np.zeros(collocation.K * self.m),) +         # u coefficients after this time interval
                                (np.zeros(N * (collocation.K + 1) * self.n),))            # xdot + x coefficients
            try:
                resmin, umin = self.solve_lp(ci, b, dext, Me, Miext, params=kwargs)
            except OptimalityError:
                umin = np.nan
            try:
                resmax, umax = self.solve_lp(-ci, b, dext, Me, Miext, params=kwargs)
            except OptimalityError:
                umax = np.nan
            uminmax[ti] = (umin, -umax)
        colltimes = np.sort(uminmax.keys())
        umin = np.array([uminmax[ti][0] for ti in colltimes])
        umax = np.array([uminmax[ti][1] for ti in colltimes])
        control_min = np.interp(times, colltimes, umin)
        control_max = np.interp(times, colltimes, umax)
        return control_min, control_max
        
    def state_variability(self, statevec, times, optval=None, **kwargs):
        """
        Solve the state variability problem at given time points.

        Arguments::
        statevec - Get variability in statevec' * x(t)
        times - list of time points at which state variability should be computed.
        optval - Objective functional value which should be achieved. If None, solve
            the optimization problem first.
        Additional arguments are passed to the ``self.solve_lp`` method.
        """
        collocation = self.params['collocation']
        optimdim = self.params["timesteps"] * (self.m * collocation.K + self.n * collocation.K + self.n)
        h = self.params["finaltime"] / self.params["timesteps"]
        N = self.params["timesteps"]
        xdotoffset = self.params["timesteps"] * self.m * collocation.K
        xoffset = xdotoffset + self.params["timesteps"] * self.n * collocation.K
        
        c,b,d,e,Me,Mi = self.discretize(collocation=self.params['collocation'])
        if optval is None:
            _, optval = self.solve_lp(c, b, d, Me, Mi, params=kwargs)
        else:
            optval = optval - e
        Miext = sparse.vstack((Mi, c)).tolil()
        dext = np.concatenate((d, np.atleast_1d(optval)))
        times = np.atleast_1d(times)
        indices = np.array([i for i in range(N) if np.any(np.abs(times - (i+1) * h) < h)])
        statevec = np.atleast_1d(statevec)
        state_min = np.zeros(len(indices))
        state_max = np.zeros(len(indices))
        for i,ind in enumerate(indices):
            ci = np.concatenate((np.zeros(xoffset),) + ind * (np.zeros(self.n),) + (statevec,) + (N-ind-1)* (np.zeros(self.n),))
            try:
                resmin, smin = self.solve_lp(ci, b, dext, Me, Miext, params=kwargs)
                state_min[i] = smin
            except OptimalityError:
                state_min[i] = np.nan
            try:
                resmax, smax = self.solve_lp(-ci, b, dext, Me, Miext, params=kwargs)
                state_max[i] = -smax
            except OptimalityError:
                state_max[i] = np.nan
        timesp = np.concatenate(([0.], h * np.atleast_1d(np.float64(indices + 1))))
        sminp = np.concatenate(([statevec.dot(self.x0)], state_min))
        smaxp = np.concatenate(([statevec.dot(self.x0)], state_max))
        return (np.interp(times, timesp, sminp), np.interp(times, timesp, smaxp))

    def solve_cvxopt(self, params={}):
        """
        Solve the LTI optimization problem with cvxopt.

        For problems with fixed terminal time, return a :py:class:`.CollocatedTrajectory`
        object representing an optimal trajectory.
        For problems with variable terminal time, return a tuple of the optimal time
        and 
        """
        warnings.warn("This function is deprecated, use 'self.solve_freeterm' instead.", DeprecationWarning)
        return self.solve_fixedterm(solver='cvxopt', params=params)

    def solve_cvxopt_fixedterm(self, params={}):
        """
        Solve the LTI optimization problem with cvxopt with fixed final time.

        If the optimization is successful, the method returns:

        * The optimal value.
        * A :py:class:`.CollocatedTrajectory` object representing an optimal trajectory.
        """
        warnings.warn("This function is deprecated, use 'self.solve_fixedterm' instead.", DeprecationWarning)
        return self.solve_fixedterm(solver='cvxopt', params=params)

    def solve_lp(self, c, b, d, Me, Mi, solver=None, params={}):
        """
        Solve the LP problem::

            min  c' x
            s.t. Me x  = b
                 Mi x <= d

        Returns an optimal solution x and objective function value c' x.
        Raises OptimalityError if solver fails.

        Available solvers with parameters:
        
        * 'cvxopt'.
        """
        solverparams = self.solverparams.copy()
        solverparams.update(params)
        if solver is None:
            solver = self.solver
        if solver == "cvxopt":
            sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvx_sparse(Mi), cvxopt.matrix(d), cvx_sparse(Me), cvxopt.matrix(b), **solverparams)
            if sol['status'] == "dual infeasible":
                raise OptimalityError("Dual infeasibility encountered.")
            if sol['status'] == "primal infeasible":
                raise OptimalityError("Primal infeasibility encountered.")
            if sol['status'] == "unknown":
                raise OptimalityError("Optimizer returned with an unknown status.")
            x = np.array(sol['x']).ravel()
            return x, sol['primal objective']
        else:
            raise ValueError("Unknown solver: %s" % solver)
            
