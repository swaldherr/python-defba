"""
Module optmetabogen.py: Dynamic optimization of metabolic-genetic networks.
"""
from __future__ import division

import cvxopt
import cvxopt.solvers
import numpy as np
import scipy.linalg as scila

def spdiag(x):
    """
    Sparse diagonal of not necessarily square matrices.
    """
    zsparse = lambda size: cvxopt.spmatrix([], [], [], size)
    xl = map(cvxopt.sparse, x)
    return cvxopt.sparse([[cvxopt.sparse([zsparse((sum(xj.size[0] for xj in xl[:i]), xi.size[1])),
                                          xi,
                                          zsparse((sum(xj.size[0] for xj in xl[(i+1):]), xi.size[1]))])]
                          for i,xi in enumerate(xl)])

class OptimalityError(Exception):
    pass

class LinOpt(object):
    """
    Dynamic optimizer for a general linear system.
    """
    def __init__(self, phi, psi, A, B, x0=0, C=None, D1=None, D2=None, umin=None, umax=None):
        if phi is not None:
            raise NotImplementedError("Optimization with phi is not implemented, use phi=None.")
        self.psi = psi
        self.A = A
        self.B = B
        self.C = None if C is not None and np.prod(np.asarray(C).shape) == 0 else C
        self.D1 = D1
        self.D2 = D2
        self.n = B.shape[0]
        self.p = B.shape[1]
        self.m = D2[0].shape[1] if D2 is not None else 0 # dimension of v(i)
        self.q1 = C.shape[0] if C is not None else 0
        self.q2 = D2[0].shape[0] if D2 is not None else 0
        self.q3 = D1[0].shape[0] if D1 is not None else 0
        self.umin = np.array(umin) if umin is not None else None
        self.umax = np.array(umax) if umax is not None else None
        if np.alltrue(x0 == 0):
            self.x0 = np.zeros((self.n, 1))
        else:
            self.x0 = np.atleast_2d(np.ravel(x0)).T

    def optimize(self, Tend, steps=10, **kwargs):
        """
        Run the dynamic optimization.

        Inputs:
        Tend - end point of the time interval
        steps - number of discretization intervals
        kwargs - options passed to the solver cvxopt.solvers.lp

        Returns:
        - list of time points
        - list of optimal control input values
        - list of state values at time points

        Raises OptimalityError if either primal or dual infeasibility occured.
        """
        DeltaT = Tend / steps
        if np.alltrue(self.A == 0):
            Ad = np.eye(self.n)
            Bd = self.B * DeltaT
        else:
            Z = np.zeros((self.p,self.n+self.p))
            expM = scila.expm(np.vstack((np.hstack((self.A * DeltaT, self.B * DeltaT)), Z)))
            Ad = np.copy(expM[:self.n,:self.n])
            Bd = np.copy(expM[:self.n,self.n:])
            assert Ad.shape == (self.n, self.n)
            assert Bd.shape == (self.n, self.p)
        self.Ad = Ad
        self.Bd = Bd
        c = np.concatenate((np.zeros(self.p * steps + self.n * (steps - 1)), -self.psi, np.zeros(self.m * steps)))
        h = cvxopt.spmatrix(-np.dot(Ad, self.x0), range(self.n), self.n * [0], (steps * (self.n + self.q1 + self.q2), 1))
        In = cvxopt.spdiag(self.n * [1.0])
        Ip = cvxopt.spdiag(self.p * [1.0])
        Im = cvxopt.spdiag(self.m * [1.0])
        Empty = cvxopt.matrix([[]], tc='d')
        zsparse = lambda size: cvxopt.spmatrix([], [], [], size)
        # H12 = cvxopt.sparse([[zsparse((self.n, self.n * i)), Ad, -In, zsparse((self.n, self.n * (steps - i - 2)))] for i in range(steps-1)] +
        #                     [-In, zsparse((self.n, self.n * (steps - 1)))])
        H12 = cvxopt.sparse([[zsparse((self.n * i, self.n)), -In, cvxopt.matrix(Ad), zsparse((self.n * (steps - i - 2), self.n))] for i in range(steps-1)] +
                            [[zsparse((self.n * (steps -1), self.n)), -In]]
            )
        H = cvxopt.sparse([[spdiag(steps * [cvxopt.matrix(Bd)]), spdiag(steps * [cvxopt.matrix(self.C)]) if self.C is not None else zsparse((0, steps * self.p)),
                            zsparse((steps * self.q2, steps * self.p))],
                           [H12, zsparse((steps * self.q1, steps * self.n)),
                            spdiag(steps * [cvxopt.matrix(self.D2[1])]) if self.D2 is not None else zsparse((0, steps * self.n))],
                           [zsparse((steps * (self.n + self.q1), steps * self.m)),
                            spdiag(steps * [-cvxopt.matrix(self.D2[0])]) if self.D2 is not None else zsparse((0, steps * self.m))]]
            )
        pmin = 0 if self.umin is None else self.p
        pmax = 0 if self.umax is None else self.p
        ng = steps * (2 * self.q3 + pmin + pmax + self.n + 2 * self.m)
        g = zsparse((ng, 1))
        if self.umin is not None:
            g[(steps * self.q3):(steps * (self.q3 + self.p)),0] = cvxopt.sparse(steps * [-cvxopt.matrix(self.umin)])
        if self.umax is not None:
            g[(steps * (self.q3 + self.p)):(steps * (self.q3 + 2 * self.p)),0] = cvxopt.sparse(steps * [cvxopt.matrix(self.umax)])
        G = cvxopt.sparse([[spdiag(steps * [-cvxopt.matrix(self.D1[1])]) if self.D1 is not None else zsparse((0, steps * self.p)),
                            spdiag(steps * [cvxopt.matrix(self.D1[1])]) if self.D1 is not None else zsparse((0, steps * self.p)),
                            spdiag(steps * [-Ip]) if pmin else zsparse((0, steps * self.p)),
                            spdiag(steps * [Ip]) if pmax else zsparse((0, steps * self.p)),
                            zsparse((steps * self.n, steps * self.p)),
                            spdiag(steps * [-cvxopt.matrix(self.D1[2])]) if self.m and self.D1 is not None else zsparse((0, steps * self.p)),
                            spdiag(steps * [cvxopt.matrix(self.D1[2])]) if self.m and self.D1 is not None else zsparse((0, steps * self.p))],
                           [spdiag(steps * [-cvxopt.matrix(self.D1[0])]) if self.D1 is not None else zsparse((0, steps * self.n)),
                            spdiag(steps * [-cvxopt.matrix(self.D1[0])]) if self.D1 is not None else zsparse((0, steps * self.n)),
                            zsparse((steps * (pmin + pmax), steps * self.n)),
                            spdiag(steps * [-In]),
                            zsparse((steps * 2 * self.m if self.D1 is not None else 0, steps * self.n))],
                           [zsparse((steps * (2 * self.q3 + pmin + pmax + self.n), steps * self.m)),
                            spdiag(steps * [-Im]) if self.m and self.D1 is not None else zsparse((0, steps * self.m)),
                            spdiag(steps * [-Im]) if self.m and self.D1 is not None else zsparse((0, steps * self.m))]]
            )
        self.lp = (c, G, g, H, h)
        self.sol = cvxopt.solvers.lp(cvxopt.matrix([[ci for ci in c]]), G, cvxopt.matrix(g), H, cvxopt.matrix(h), **kwargs)
        if self.sol["dual infeasibility"] is None:
            raise OptimalityError("Dual infeasibilty encountered.")
        if self.sol["primal infeasibility"] is None:
            raise OptimalityError("Primal infeasibilty encountered.")
        times = [i*DeltaT for i in range(1,steps+1)]
        u = [np.array(self.sol['x']).ravel()[i*self.p:(i+1)*self.p] for i in range(steps)]
        x = [np.array(self.sol['x']).ravel()[(steps*self.p+i*self.n):(steps*self.p+(i+1)*self.n)] for i in range(steps)]
        return times, u, x

def rocket_car_test():
    """
    Test dynamic optimizer with the rocket car example.
    """
    A = np.array([[0, 1],[0, 0]])
    B = np.array([[0],[1]])
    lo = LinOpt(None, np.array([1.0, 0.0]), A, B, umin=np.array([-1]), umax=[1])
    return lo

class Metabogen(object):
    """
    Dynamic optimizer for a metabolic-genetic network.
    """
    def __init__(self, net, ext_species, metab_species, gen_species, enzymes=None, catconstants=1.0, alpha=1.0, epsilon=1.0, **kwd):
        self.alpha0 = alpha * epsilon
        extind = np.array([net.species.index(x) for x in ext_species], dtype=np.int)
        metabind = np.array([net.species.index(x) for x in metab_species], dtype=np.int)
        genind = np.array([net.species.index(x) for x in gen_species], dtype=np.int)
        # exchange reactions: everything where ext_species is involved
        self.vext = []
        self.vgen = []
        self.vmetab = []
        for i,v in enumerate(net.reactions):
            if not np.alltrue(net.stoich[extind,i] == 0):
                self.vext.append(v)
            elif not np.alltrue(net.stoich[genind,i] == 0):
                self.vgen.append(v)
            else:
                self.vmetab.append(v)
        self.reactions = self.vext + self.vmetab + self.vgen
        vextind = np.array([net.reactions.index(v) for v in self.vext], dtype=np.int)
        vmetabind = np.array([net.reactions.index(v) for v in self.vmetab], dtype=np.int)
        vgenind = np.array([net.reactions.index(v) for v in self.vgen], dtype=np.int)
        self.Syy = -net.stoich[np.ix_(extind,vextind)]
        self.Sxy = net.stoich[np.ix_(metabind,vextind)]
        self.Sxx = net.stoich[np.ix_(metabind,vmetabind)]
        self.Sxp = -net.stoich[np.ix_(metabind,vgenind)] / alpha
        self.Spy = net.stoich[np.ix_(genind,vextind)]
        self.Spp = net.stoich[np.ix_(genind,vgenind)]
        self.ext_species = ext_species
        self.metab_species = metab_species
        self.gen_species = gen_species
        self.dynspecies = self.ext_species + self.gen_species
        self.numreact = len(self.reactions)
        self.yp0 = net.eval(self.ext_species + self.gen_species)
        self.enzymes = enzymes
        if type(catconstants) is dict:
            self.catconstants = dict.fromkeys(self.vext + self.vgen + self.vmetab, 1.0)
            self.catconstants.update(catconstants)
        else:
            self.catconstants = dict.fromkeys(self.vext + self.vgen + self.vmetab, catconstants)
        self.set_constraints(**kwd)

    def set_constraints(self, umin=None, umax=None, bvec=None, y0=None, p0=None):
        if bvec is None:
            Sxpsum = np.sum(self.Sxp, 0)
            self.bvec = np.dot(self.Spp, Sxpsum)
        else:
            self.bvec = bvec
        if umin is not None:
            self.umin = np.asarray(umin) if np.iterable(umin) else umin * np.ones(self.numreact)
        if umax is not None:
            self.umax = np.asarray(umax) if np.iterable(umin) else umax * np.ones(self.numreact)

    def optimize(self, Tend, steps=10, **kwargs):
        """
        Run the dynamic optimization to Tend with initial condition [y0, p0].
        """
        psi = np.concatenate((np.zeros(len(self.ext_species)), self.bvec))
        A = 0
        B = np.vstack((np.hstack((-self.Syy, np.zeros((len(self.ext_species), len(self.vmetab) + len(self.vgen))))),
                       np.hstack((self.Spy, np.zeros((len(self.gen_species), len(self.vmetab))), self.alpha0 * self.Spp))))
        C = np.hstack((self.Sxy, self.Sxx, -self.alpha0 * self.Sxp))
        react_sharedenz = sum((list(self.enzymes[p]) for p in self.enzymes if len(self.enzymes[p]) > 1), [])
        shared_enz = [p for p in self.enzymes if len(self.enzymes[p]) > 1]
        single_enz = [p for p in self.enzymes if len(self.enzymes[p]) == 1]
        D11 = np.zeros((len(single_enz), len(self.dynspecies)))
        D12 = np.zeros((len(single_enz), self.numreact))
        for i,p in enumerate(single_enz):
            D11[i,self.dynspecies.index(p)] = 1.0
            D12[i,self.reactions.index(self.enzymes[p][0])] = 1.0 / self.catconstants[self.enzymes[p][0]]
        D13 = np.zeros((len(shared_enz), self.numreact))
        D21 = np.eye(len(shared_enz))
        D22 = np.zeros((len(shared_enz), len(self.dynspecies)))
        for i,p in enumerate(shared_enz):
            for v in self.enzymes[p]:
                D13[i,self.reactions.index(v)] = 1.0 / self.catconstants[v]
                D22[i,self.dynspecies.index(p)] = 1.0
        D1 = (D11, D12, D13)
        if len(shared_enz):
            D2 = (D21, D22)
        else:
            D2 = None
        self.lo = LinOpt(None, psi, A, B, x0=self.yp0, C=C, D1=D1, D2=D2, umin=self.umin, umax=self.umax)
        return self.lo.optimize(Tend, steps=steps, **kwargs)

def test_minimal_net():
    """
    Test metabolic-genetic optimization with a minimal network.
    """
    import brn
    species = ["Y", "X", "E", "R"]
    reactions = ["v0", "venz", "vrib"]
    stoich = [[-1, 0, 0],[1, -200, -500],[0, 1, 0],[0, 0, 1]]
    net = brn.Brn(stoich, species, reactions)
    net.set(dict(zip(species, [100.0, 0, 1.0, 1.0])))
    mg = Metabogen(net, ["Y"], ["X"], ["E", "R"], enzymes={"E": ("v0",), "R": ("venz", "vrib")}, alpha=100, epsilon=0.01)
    mg.set_constraints(umin=[0,0,0])
    return mg
