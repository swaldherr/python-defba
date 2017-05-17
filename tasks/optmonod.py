"""
Optimize minimal metabolic-genetic (Monod) network with MGOpt
"""
# Time-stamp: <Last change 2013-06-19 16:47:26 by Steffen Waldherr>

import numpy as np
from scipy import integrate

import scripttool
import brn

from src import metabogen, linopt

def mmg_net():
    species = ["Y", "X", "P"]
    reactions = ["vup", "vp"]
    stoich = [[-1.0, 0.0],[1.0, -10.],[0., 1.]]
    net = brn.Brn(stoich, species, reactions)
    net.set({"Y": 100.0, "X":0.0, "P":1.0})
    net.set({"vup": "0.5 * Y * P/(1 + Y)",
             "vp": "0.05 * X * P/(0.01 + X)"})
    return net

def monod_net():
    net = mmg_net()
    mg = metabogen.Metabogen(net, ["Y"], ["X"], ["P"], enzymes={"P": ("vup", "vp")}, catconstants={"vup":1.0, "vp":0.1})
    mg.set_parameters(collocation=linopt.LRrevCollocation(3))
    mg.set_reaction_bounds(vmin=0)
    return mg

@scripttool.memoize.filecache
def mgoptimize(mg):
    return mg.optimize()

@scripttool.memoize.filecache
def mg_concentration_variability(mg, times, obj=None, optparams={}):
    return mg.concentration_variability("P", times, optval=obj, optimizer='cvxopt', optparams=optparams)

@scripttool.memoize.filecache
def mg_flux_variability(mg, times, obj=None, optparams={}):
    return mg.flux_variability("vup", times, optval=obj, optimizer='cvxopt', optparams=optparams)

@scripttool.memoize.filecache
def mgoptimize_discount(mg, C0, discount, finaltime, timesteps):
    C = lambda t: C0 * np.exp(-discount * t)
    mg.set_objective(C, np.zeros((2,)), 0, np.zeros((2,)))
    res = mg.optimize(finaltime=finaltime, timesteps=timesteps)
    # set objective to a pickle-able value again for caching
    mg.set_objective(C0, np.zeros((2,)), 0, np.zeros((2,)))
    return res

@scripttool.memoize.filecache
def losolve(o, solver):
    return o.solve(solver=solver)

class SimulateMonod(scripttool.Task):
    """
    Simulation of the Monod growth model
    """
    customize = {"rho":0.1,
                 "mumax": 0.05,
                 "Ky_vole": 1,
                 "Kx_volc": 0.01,
                 "alpha": 10,
                 }
    def monod(self, x, t):
        return [- 1.0 / self.rho * self.mumax * x[0] / (self.Ky_vole + x[0]) * x[1],
                self.mumax * x[0] / (self.Ky_vole + x[0]) * x[1]]
    
    def run(self):
        time = np.arange(80.0, step=80.0/200)
        x = integrate.odeint(self.monod, np.array([100.0, 1.0]), time)
        net = mmg_net()
        t,xnet = net.simulate(time)
        xnet = np.asarray(xnet)
        fig, ax = self.make_ax(name="monod-growth-kinetic",
                               xlabel="Time [min]",
                               ylabel="Mass [relative dw]",
                               figtype="small")
        ax.semilogy(time, x[:,0], '--g', basey=2)
        ax.semilogy(time, self.alpha * x[:,1], 'r', lw=2, basey=2)
        ax.semilogy(time, self.alpha * xnet[:,2], 'r:', basey=2)
        ax.set_ylim(5, 150)

class SimulateMGNetwork(scripttool.Task):
    """
    Simulation of a minimal metabolic-genetic network with Monod kinetics.
    """
    def run(self):
        time = np.arange(80.0, step=80.0/200)
        net = mmg_net()
        t,x = net.simulate(time)
        x = np.asarray(x)
        fig, ax = self.make_ax(name="minmg-simresult",
                               xlabel="Time [min]",
                               ylabel="Mass [relative]",
                               figtype="small")
        ax.semilogy(time, x[:,0], '--g', basey=2)
        ax.semilogy(time, 10 * x[:,2], 'r', lw=2, basey=2)
        ax.set_ylim(5, 150)

def plot_monod_optim(task, traj, tf, obj, svar=None, fvar=None):
    """
    Plotting routine for evaluation of Monod network optimization results.
    """
    time = np.arange(tf, step=tf/200)
    r = traj.get_fluxes(time)
    x = traj.get_substances(time)
    title = r"Maximization of $%s$" % obj
    fig, ax = task.make_ax(name="monod-fluxes-%s" % obj.replace('_', ''),
                           xlabel="Time [min]",
                           ylabel="V [1/min]",
                           figtype="small")
    if fvar is not None:
        # ax.fill_between(fvar[0], fvar[1] / traj.get_substances(fvar[0])[:,1], fvar[2] / traj.get_substances(fvar[0])[:,1], alpha=0.3, color='g')
        ax.fill_between(fvar[0], fvar[1], fvar[2], alpha=0.3, color='g')
    ax.plot(time, r[:,0], 'g')
    ax.plot(time, r[:,1] , '--r')
    ax.set_ylim(0,6)
    ax.set_xlim(0, 80)
    fig1, ax1 = task.make_ax(name="monod-concentrations-%s" % obj.replace('_', ''),
                             xlabel="Time [min]",
                             ylabel="Mass [relative dw]",
                             figtype="small")
    if svar is not None:
        ax1.fill_between(svar[0], traj.mg.alpha * svar[1], traj.mg.alpha * svar[2], alpha=0.3, color='r')
        # ax1.semilogy(svar[0], svar[1], '--r')
        # ax1.semilogy(svar[0], svar[2], '--r')
    ax1.semilogy(time, x[:,0], '--g', basey=2)
    ax1.semilogy(time, traj.mg.alpha * x[:,1], 'r', lw=2, basey=2)
    ax1.set_ylim(5, 150)
    ax1.set_xlim(0, 80)
    fig2, ax2 = task.make_ax(name="monod-constraints-%s" % obj.replace('_', ''),
                             xlabel="Time [min]",
                             ylabel="Constraints",
                             figtype="small")
    c = traj.get_constraints(time)
    # Positivity of Y
    ax2.plot(time, c[:,0])
    # Enzymatic constraint
    ax2.plot(time, c[:,4])
    ax2.set_xlim(0, 80)
    ax2.set_ylim(-100, 20)
    return fig, ax, fig1, ax1, fig2, ax2

class OptMonodDiscount(scripttool.Task):
    """
    Metabogen optimization with a simple Monod network, discounted objective functional.
    """
    customize = {"discount":0.01}

    def run(self):
        mg = monod_net()
        discountedC = metabogen.DiscountedVector(-np.array([0, 1.0]), self.discount)
        mg.set_objective(discountedC, np.zeros((2,)), 0.0, np.zeros((2,)))
        mg.set_parameters(finaltime=80.0, timesteps=50)
        tf, obj, traj = mgoptimize(mg)
        self.printf("Optimal objective function value: %g" % obj)
        print "Optimal objective function value: %g" % obj
        ts = (mg.alpha / mg.catconstants["vup"] + 1. / mg.catconstants["vp"]) * np.log(1 + mmg_net().eval("Y/P") / mg.alpha)
        self.printf("Analytical switch time ts = %g" % ts)
        timess = np.linspace(0, 80, num=11)
        smin, smax = mg_concentration_variability(mg, timess, optparams={'solver':'glpk'})
        self.printf("Minimum P: %s" % str(smin))
        self.printf("Maximum P: %s" % str(smax))
        timesf = np.linspace(0, 80, num=11)
        fmin, fmax = mg_flux_variability(mg, timesf, optparams={'solver':'glpk'})
        self.printf("Minimum flux vup: %s" % str(fmin))
        self.printf("Maximum flux vup: %s" % str(fmax))
        plot_monod_optim(self, traj, tf, "J_2", svar=(timess, smin, smax))#, fvar=(timesf, fmin, fmax))
        
class OptMonod(scripttool.Task):
    """
    Metabogen optimization with a simple Monod network.
    """
    customize = {"alpha":10}

    def run(self):
        species = ["Y", "X", "P"]
        reactions = ["vup", "vp"]
        stoich = [[-1, 0],[1, -10],[0, 1]]
        net = brn.Brn(stoich, species, reactions)
        net.set(dict(zip(species, [100.0, 0, 1.0])))
        mg = monod_net()
        ts = (mg.alpha / mg.catconstants["vup"] + 1. / mg.catconstants["vp"]) * np.log(1 + mmg_net().eval("Y/P") / mg.alpha)
        self.printf("Analytical switch time ts = %g" % ts)
        mg.set_reaction_bounds(vmin=0)
        mg.set_objective(np.zeros((2,)), np.zeros((2,)), 0, -np.array([0, 1.]))
        mg.set_parameters(finaltime=80.0, timesteps=50)
        tf, obj, traj = mgoptimize(mg)
        self.printf("Optimal objective function value: %g" % obj)
        timess = np.linspace(0, 80, num=11)
        smin, smax = mg_concentration_variability(mg, timess, obj)
        timesf = np.linspace(0, 80, num=11)
        fmin, fmax = mg_flux_variability(mg, timesf, optparams={'solver':'glpk'})
        self.printf("Minimum flux vup: %s" % str(fmin))
        self.printf("Maximum flux vup: %s" % str(fmax))
        fig, ax, fig1, ax1, fig2, ax2 = plot_monod_optim(self, traj, tf, "J_1", svar=(timess, smin, smax), fvar=(timesf, fmin, fmax))
        
        o = mg.get_optimization_problem()
        o.set_parameters(finaltime=80.0, timesteps=50, fixedterm=False, finaltimetolerance=0.1)
        o.add_terminal_constraint(np.array([1.0, 0]), 0)
        o.set_objective(np.zeros((2,)), np.zeros((2,)), 1, np.zeros(2))
        tf, obj, sol = losolve(o, solver='cvxopt')
        traj = metabogen.MGTrajectory(mg, sol)
        self.printf("Optimal objective function value 2: %g" % obj)
        self.printf("Terminal time 2: %g" % tf)
        plot_monod_optim(self, traj, tf, "J_3")

# creation of my experiments
# scripttool.register_task((taskoption=["custom"]), ident="task_custom")
scripttool.register_task(OptMonod(), ident="optmonod")
scripttool.register_task(OptMonodDiscount(), ident="optmonod-discount")
scripttool.register_task(SimulateMonod(), ident="monod-kinetic")
scripttool.register_task(SimulateMGNetwork(), ident="minmg-simulation")
