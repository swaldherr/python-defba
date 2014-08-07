"""
Module metabogen.py: Dynamic optimization of metabolic-genetic networks.
"""
# Time-stamp: <Last change 2013-06-20 09:58:43 by Steffen Waldherr>
from __future__ import division

from itertools import product

import numpy as np
from numpy import linalg
import sympy

import cvxopt
import cvxopt.solvers

import brn

from src import linopt

def unitvector(n, i):
    uv = np.zeros(n)
    uv[i] = 1.0
    return uv

class MGTrajectory(linopt.CollocatedTrajectory):
    def __init__(self, mg, trajdata):
        super(MGTrajectory, self).__init__(trajdata)
        self.mg = mg

    def get_substances(self, t):
        x = self.get_state(t)
        return linopt.block_diagonal((np.eye(len(self.mg.ext_species)), np.eye(len(self.mg.gen_species)) / self.mg.alpha)).dot(x.T).T

    def get_fluxes(self, t):
        u = self.get_control(t)
        return self.mg.Uscaled.dot(u.T).T

    def get_constraints(self, t):
        z = self.get_substances(t)
        v = self.get_fluxes(t)
        clist = self.mg.get_constraints()
        cdim = sum((len(Gk) for Gz, Gv, Gk in clist))
        c = np.zeros((len(t), cdim))
        inda = 0
        for Gz, Gv, Gk in clist:
            indb = inda + len(Gk)
            for i,ti in enumerate(t):
                c[i,inda:indb] = Gz.dot(z[i,:]) + Gv.dot(v[i,:]) + Gk
            inda = indb
        return c

class DiscountedVector(object):
    def __init__(self, vec0, discount=0):
        self.vec0 = vec0
        self.discount = discount

    def __call__(self, t):
        return self.vec0 * np.exp(- self.discount * t)
        
class Metabogen(object):
    """
    Dynamic optimizer for a metabolic-genetic network.
    """
    def __init__(self, net, ext_species, metab_species, gen_species, ext_react=[], enzymes=None, catconstants=1.0, alpha=0):
        excind = np.array([net.species.index(x) for x in ext_species], dtype=np.int)
        metabind = np.array([net.species.index(x) for x in metab_species], dtype=np.int)
        genind = np.array([net.species.index(x) for x in gen_species], dtype=np.int)
        # exchange reactions: everything where ext_species is involved
        self.vexc = []
        self.vgen = []
        self.vmetab = []
        self.vext = ext_react
        for i,v in enumerate(net.reactions):
            if not np.alltrue(net.stoich[excind,i] == 0) and v not in self.vext:
                self.vexc.append(v)
            elif not np.alltrue(net.stoich[genind,i] == 0):
                self.vgen.append(v)
            else:
                self.vmetab.append(v)
        self.reactions = self.vexc + self.vmetab + self.vgen
        vexcind = np.array([net.reactions.index(v) for v in self.vexc], dtype=np.int)
        vmetabind = np.array([net.reactions.index(v) for v in self.vmetab], dtype=np.int)
        vgenind = np.array([net.reactions.index(v) for v in self.vgen], dtype=np.int)
        vextind = np.array([net.reactions.index(v) for v in self.vext], dtype=np.int)
        self.Syy = -net.stoich[np.ix_(excind,vexcind)]
        self.Sy0 = net.stoich[np.ix_(excind,vextind)]
        self.Sxy = net.stoich[np.ix_(metabind,vexcind)]
        self.Sxx = net.stoich[np.ix_(metabind,vmetabind)]
        Sxp = -net.stoich[np.ix_(metabind,vgenind)]
        if alpha == 0:
            alpha = np.max(np.abs(Sxp))
        self.alpha = float(alpha) if (alpha != 0) else 1.0
        self.Sxp = Sxp / self.alpha
        self.Spy = net.stoich[np.ix_(genind,vexcind)]
        self.Spp = net.stoich[np.ix_(genind,vgenind)]
        self.Sp0 = net.stoich[np.ix_(genind,vextind)]
        self.ext_species = ext_species
        self.metab_species = metab_species
        self.gen_species = gen_species
        self.dynspecies = self.ext_species + self.gen_species
        self.numreact = len(self.reactions)
        self.z0 = net.eval(self.ext_species + self.gen_species)
        self.z0[len(self.ext_species):] = self.alpha * self.z0[len(self.ext_species):]
        self.enzymes = enzymes
        if type(catconstants) is dict:
            self.catconstants = dict.fromkeys(self.vexc + self.vgen + self.vmetab, 1.0)
            self.catconstants.update(catconstants)
        else:
            self.catconstants = dict.fromkeys(self.vexc + self.vgen + self.vmetab, catconstants)
        self.set_reaction_bounds(-np.inf, np.inf)
        self.constraints = []

        if len(self.vext) > 0:
            jacext = np.asarray(net.jacobian(self.vext, self.ext_species + self.gen_species)(net.eval(net.species), net.eval(net.parameters)))
            ext0 = net.eval(self.vext) - jacext.dot(self.z0)
            # test linearity
            zsym = np.array(sympy.symbols(self.ext_species + self.gen_species))
            vexpected = jacext.dot(zsym) + ext0
            vactual = np.array(net.eval(self.vext, valuedict=dict(zip(self.ext_species + self.gen_species, zsym)), typestrict=False))
            if np.any(vexpected - vactual >= np.MachAr().eps):
                raise ValueError("The network does not have affine extracellular reactions: vext = %s" % str([net.values[v] for v in self.vext]))
            self.A = np.vstack((self.Sy0, self.Sp0)).dot(jacext)
            self.A0 = np.vstack((self.Sy0, self.Sp0)).dot(ext0)
        else:
            self.A = np.zeros((len(self.z0), len(self.z0)))
            self.A0 = np.zeros(len(self.z0))

        # decomposition of the stoichiometric matrix for metabolic species to get projection on flux space
        self.Sx = np.hstack((self.Sxy, self.Sxx, -self.Sxp))
        U, sv, V = linalg.svd(self.Sx)
        maxabs = np.max(sv)
        maxdim = max(self.Sx.shape)
        tol = maxabs * maxdim * np.MachAr().eps
        rankS = np.sum(sv > tol)
        self.W = V.T[:, :rankS]
        self.U = V.T[:, rankS:]
        self.Uscaled = linopt.block_diagonal((np.eye(len(self.vexc)),np.eye(len(self.vmetab)),np.eye(len(self.vgen))/self.alpha)).dot(V.T[:, rankS:])
        self.B = np.vstack((-self.Syy.dot(self.U[:len(self.vexc), :]), self.Spp.dot(self.U[(len(self.vexc)+len(self.vmetab)):, :])))
        self.S = np.vstack((np.hstack((-self.Syy, np.zeros((len(self.ext_species), len(self.vmetab))), np.zeros((len(self.ext_species), len(self.vgen))))),
                            np.hstack((self.Sxy, self.Sxx, -Sxp)), # using unscaled version of Sxp here!
                            np.hstack((np.zeros((len(self.gen_species), len(self.vexc))), np.zeros((len(self.gen_species), len(self.vmetab))), self.Spp))
            ))
        self.params = {"timesteps": 50,
                       "finaltime": 1.0,
                       "fixedterm": True,
                       "collocation": linopt.Collocation(3, 'lagrange', 'radau'),
                       "finaltimetolerance": 0.01
                       }
        self.solverparams = {}

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

    def set_objective(self, C, D, E, F):
        """
        Set the objective functional of the dynamic optimization.

        The optimization minimizes the functional J defined as::

            J = int_0^{t_f} (C(t) z(t) + D(t) v(t) + E(t)) dt + F z(t_f)
        """
        self.C, self.D, self.E, self.F = C, D, E, F
        if type(self.C) is DiscountedVector:
            vec0 = self.C.vec0.copy()
            vec0[len(self.ext_species):] = vec0[len(self.ext_species):] / self.alpha
            self.C = DiscountedVector(vec0, self.C.discount)
        else:
            self.C[len(self.ext_species):] = self.C[len(self.ext_species):] / self.alpha
        self.F[len(self.ext_species):] = self.F[len(self.ext_species):] / self.alpha

    def set_initial_condition(self, y0=None, p0=None):
        """
        Set the initial condition to (y0, p0).
        """
        if y0 is not None:
            self.z0[:len(self.ext_species)] = y0
        if p0 is not None:
            self.z0[len(self.ext_species):] = self.alpha * p0

    def set_reaction_bounds(self,  vmin=None, vmax=None):
        """
        Set lower and upper reaction bounds for this network.
        """
        if vmin is not None:
            self.vmin = np.fromiter(vmin, dtype=np.float64) if np.iterable(vmin) else vmin * np.ones(self.numreact)
            self.vmin[len(self.ext_species):] = self.alpha * self.vmin[len(self.ext_species):]
        if vmax is not None:
            self.vmax = np.fromiter(vmax, dtype=np.float64) if np.iterable(vmin) else vmax * np.ones(self.numreact)
            self.vmax[len(self.ext_species):] = self.alpha * self.vmax[len(self.ext_species):]

    def add_constraint(self, Gz, Gv, Gk):
        """
        Add a path constraint of the type

            Gz z(t) + Gv v(t) + Gk <= 0.

        Note that species positivity, upper / lower reaction bounds and enzymatic constraints are generated automatically
        and need not be added through this function.
        """
        self.constraints.append((Gz, Gv, Gk))

    def get_optimization_problem(self, params={}):
        o = linopt.LinOpt(self.A, self.B, self.z0, self.A0)
        o.set_parameters(**self.params)
        o.set_parameters(**self.solverparams)
        o.set_parameters(**params)
        o.set_objective(self.C, self.D.dot(self.U), self.E, self.F)
        for Gz, Gv, Gk in self.get_constraints():
            Gzs = Gz.dot(linopt.block_diagonal((np.eye(len(self.ext_species)), np.eye(len(self.gen_species))/self.alpha)))
            Gu = Gv.dot(self.Uscaled)
            scale = max((np.max(np.abs(i)) for i in (Gzs, Gu, Gk)))
            if scale > 0:
                o.add_path_constraint(Gzs / scale, Gu / scale, Gk / scale)
        return o

    def get_enzyme_constraints(self):
        econstr = []
        offset = len(self.ext_species)
        for e in self.enzymes:
            ind = [self.reactions.index(v) for v in self.enzymes[e]]
            posind = [i for i in ind if self.vmin[i] >= 0]
            negind = [i for i in ind if self.vmax[i] <= 0]
            indset = set(posind).union(set(negind))
            mixedind = [i for i in ind if i not in indset]
            ke = 2 ** len(mixedind)
            Hce = np.zeros((ke, self.numreact), dtype=np.float64)
            Hee = np.zeros((ke, len(self.dynspecies)), dtype=np.float64)
            inde = self.gen_species.index(e)
            for j,signpattern in enumerate(product(*tuple((1.0,) if i in posind else (-1.0,) if i in negind else (1.0, -1.0) for i in ind))):
                for k,v,s in zip(ind,self.enzymes[e],signpattern):
                    if v in self.vgen:
                        Hce[j,k] = s / self.catconstants[v]
                    else:
                        Hce[j,k] = s * self.alpha / self.catconstants[v]
                    Hee[j, offset + inde] = -1
            econstr.append((Hee, Hce, np.zeros(Hee.shape[0])))
        return econstr

    def get_constraints(self):
        """
        Generate a list of biophysical constraints on the network.

        Return a list of tuples (Gz, Gv, Gk) corresponding to constraints of the form
            Gz z + Gv v + Gk <= 0.

        Raises ValueError if there is i such that vmin[i] == vmax[i].
        """
        constraints = []
        constraints.append((-np.eye(len(self.dynspecies)), np.zeros((len(self.dynspecies), self.numreact)), np.zeros(len(self.dynspecies))))
        for i, (vmin, vmax) in enumerate(zip(self.vmin, self.vmax)):
            if vmin == vmax:
                raise ValueError("vmin[%d] == vmax[%d] == %g. Use an equality constraint instead and put -inf, inf here!" % (i, i, vmin))
            if np.isfinite(vmin):
                constraints.append((np.zeros((1, len(self.dynspecies))), -np.atleast_2d(unitvector(self.numreact, i)), -np.atleast_1d(vmin)))
            if np.isfinite(vmax):
                constraints.append((np.zeros((1, len(self.dynspecies))), np.atleast_2d(unitvector(self.numreact, i)), np.atleast_1d(vmax)))
        # enzymatic constraints
        offset = len(self.ext_species)
        for e in self.enzymes:
            ind = [self.reactions.index(v) for v in self.enzymes[e]]
            posind = [i for i in ind if self.vmin[i] >= 0]
            negind = [i for i in ind if self.vmax[i] <= 0]
            indset = set(posind).union(set(negind))
            mixedind = [i for i in ind if i not in indset]
            ke = 2 ** len(mixedind)
            Hce = np.zeros((ke, self.numreact), dtype=np.float64)
            Hee = np.zeros((ke, len(self.dynspecies)), dtype=np.float64)
            inde = self.gen_species.index(e)
            for j,signpattern in enumerate(product(*tuple((1.0,) if i in posind else (-1.0,) if i in negind else (1.0, -1.0) for i in ind))):
                for k,v,s in zip(ind,self.enzymes[e],signpattern):
                    Hce[j,k] = s / self.catconstants[v]
                    Hee[j, offset + inde] = -1
            constraints.append((Hee, Hce, np.zeros(Hee.shape[0])))
        for Gz, Gv, Gk in self.constraints:
            constraints.append((Gz, Gv, Gk))
        return constraints

    def optimize(self, optimizer='cvxopt', **kwargs):
        """
        Run the dynamic optimization.
        """
        lo = self.get_optimization_problem(kwargs)
        lo.set_solver(optimizer)
        if lo.params['fixedterm']:
            obj, sol = lo.solve()
            tf = lo.params['finaltime']
        else:
            tf, obj, sol = lo.solve()
        return tf, obj, MGTrajectory(self, sol)

    def concentration_variability(self, specie, times, optval=None, optimizer='cvxopt', optparams={}, **kwargs):
        """
        Solve the species variability problem at given time points.

        Arguments::
        specie - Name of specie to use in variability analysis
        times - list of time points at which concentration variability should be computed.
        optval - Objective functional value which should be achieved. If None, solve
            the optimization problem first.
        Additional arguments are passed to the ``self.get_optimization_problem`` method.
        """
        lo = self.get_optimization_problem(kwargs)
        lo.set_solver(optimizer)
        statevec = np.zeros(len(self.dynspecies))
        statevec[self.dynspecies.index(specie)] = 1
        xmin, xmax = lo.state_variability(statevec, times, optval=optval, **optparams)
        smin = xmin / self.alpha if specie in self.gen_species else xmin
        smax = xmax / self.alpha if specie in self.gen_species else xmax
        return smin, smax
        
    def flux_variability(self, flux, times, optval=None, optimizer='cvxopt', optparams={}, **kwargs):
        """
        Solve the flux variability problem at given time points.

        Arguments::
        flux - Name of reaction to use in variability analysis
        times - list of time points at which flux variability should be computed.
        optval - Objective functional value which should be achieved. If None, solve
            the optimization problem first.
        Additional arguments are passed to the ``self.get_optimization_problem`` method.
        """
        lo = self.get_optimization_problem(kwargs)
        lo.set_solver(optimizer)
        fluxvec = self.Uscaled[self.reactions.index(flux), :]
        assert len(fluxvec) == lo.B.shape[1]
        vmin, vmax = lo.control_variability(fluxvec, times, optval=optval, **optparams)
        return vmin, vmax
    
    def rba(self, mu=1., tolerance=None, **kwargs):
        """
        Solve the static RBA problem with optimal growth rate by bisection.

        Returns muopt, (vexc, vmetab, vgen, p).

        See rba_mu() for optional arguments.
        """
        testfun = lambda mutest: self.rba_mu(mu=mutest, **kwargs)
        return linopt.bisection(testfun, mu, **dict([("tolerance", tolerance)] if tolerance is not None else []))

    def rba_mu(self, mu=0.0, optimizer='cvxopt', bmvec=None, biomass=1.0, **kwargs):
        """
        Solve the static RBA problem for fixed growth rate mu.

        If bmvec is not None add the constraint that bmvec . x = biomass

        Returns (True, (vexc, vmetab, vgen, p)) if feasible.
        Returns (False, None) if not feasible.

        Additional arguments are passed to the LP solver.
        """
        Me = np.hstack((self.S[-len(self.gen_species):,:].dot(self.Uscaled), -np.eye(len(self.gen_species)) * mu / self.alpha))
        b = np.zeros(Me.shape[0])
        c = np.zeros(Me.shape[1])
        # add path constraints for uptake reactions without substrate
        vminorig = self.vmin.copy()
        vmaxorig = self.vmax.copy()
        for i, vy in enumerate(self.vexc):
            for j, Sji in enumerate(self.Syy[:,i]):
                if Sji > 0 and self.z0[j] <= 0 and self.vmax[i]>0:
                    self.vmax[i] = 0
                if Sji < 0 and self.z0[j] <= 0 and self.vmin[i]<0:
                    self.vmin[i] = 0
        for i, (vmin, vmax) in enumerate(zip(self.vmin, self.vmax)):
            if vmin == vmax:
                self.vmin[i] = -np.inf
                self.vmax[i] = np.inf
                Me = np.vstack(( Me, np.hstack(( np.atleast_2d(unitvector(self.numreact, i)).dot(self.Uscaled), np.zeros((1, len(self.gen_species))) )) ))
                b = np.concatenate(( b, [0.]))
        if bmvec is not None:
            Me = np.vstack(( Me, np.concatenate(( np.zeros(self.Uscaled.shape[1]), bmvec)) ))
            b = np.concatenate(( b, [self.alpha * biomass] ))
        o = self.get_optimization_problem()
        self.vmin = vminorig
        self.vmax = vmaxorig
        Mi = np.vstack(tuple(np.hstack((Gui, Gxi[:,len(self.ext_species):])) for (Gxi, Gui, Gki) in o.path_constraints
                             if np.max(np.abs(np.hstack((Gui, Gxi[:,len(self.ext_species):])))) > 0))
        d = np.concatenate(tuple(-Gki for (Gxi, Gui, Gki) in o.path_constraints if np.max(np.abs(np.hstack((Gui, Gxi[:,len(self.ext_species):])))) > 0))
        sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(Mi), cvxopt.matrix(d), cvxopt.matrix(Me), cvxopt.matrix(b), **kwargs)
        if sol['status'] == "dual infeasible":
            raise linopt.OptimalityError("Dual infeasibility encountered.")
        if sol['status'] == "unknown":
            raise linopt.OptimalityError("Optimizer returned with an unknown status.")
        if sol['status'] == "primal infeasible":
            return (False, None)
        x = np.array(sol['x']).ravel()
        v = self.Uscaled.dot(x[:self.Uscaled.shape[1]])
        p = x[-len(self.gen_species):] / self.alpha
        return (True, tuple(np.split(v, [len(self.vexc), len(self.vexc)+len(self.vmetab)])) + (p,))
                
def test_monod_net():
    import brn
    species = ["Y", "X", "P"]
    reactions = ["vup", "vp"]
    stoich = [[-1, 0],[1, -10],[0, 1]]
    net = brn.Brn(stoich, species, reactions)
    net.set(dict(zip(species, [100.0, 0, 1.0])))
    mg = Metabogen(net, ["Y"], ["X"], ["P"], enzymes={"P": ("vup", "vp")})
    mg.set_reaction_bounds(vmin=0)
    mg.set_objective(np.zeros((2,)), np.zeros((2,)), 0, -np.array([0, 1.0]))
    return mg

def test_er_net():
    """
    Test metabolic-genetic optimization with a enzyme-ribosome network.
    """
    import brn
    species = ["Y", "X", "E", "R"]
    reactions = ["v0", "venz", "vrib"]
    stoich = [[-1, 0, 0],[1, -200, -500],[0, 1, 0],[0, 0, 1]]
    net = brn.Brn(stoich, species, reactions)
    net.set(dict(zip(species, [100.0, 0, 1.0, 1.0])))
    mg = Metabogen(net, ["Y"], ["X"], ["E", "R"], enzymes={"E": ("v0",), "R": ("venz", "vrib")})
    mg.set_constraints(vmin=[0,0,0])
    return mg
