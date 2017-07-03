"""
Optimize metabolic-genetic network of core metabolism example.
"""
# Time-stamp: <Last change 2016-09-17 18:35:11 by Steffen Waldherr>
import os
import csv
import shelve
from itertools import combinations

import numpy as np
from scipy import integrate

import scripttool
import brn

from src import metabogen
from src import linopt
from src import coremetabogen

scenarios = {
    'carbon-switch': dict(Carb1=5, Carb2=10, O2ext=20),
    'carbon2-switch': dict(Carb1=0, Carb2=15, O2ext=20),
    'oxygen-switch': dict(Carb1=0, Carb2=15, O2ext=2),
    'aa-switch': dict(Carb1=10, Carb2=0, O2ext=20, Hext=5),
    'rich-medium': dict(Carb1=50, Carb2=0, O2ext=8, Hext=5, Fext=5),
    'aerobic-anaerobic': dict(Carb1=50, Carb2=0, O2ext=10),
    }

scen_enzymes = dict(((k, []) for k in scenarios.keys()))
scen_enzymes['carbon-switch'].extend(['ETc1', 'ETc2', 'ER2', 'ER3'])

pre_initials = dict(((k, {}) for k in scenarios.keys()))

optparameters = dict(
    finaltime=30.,
    timesteps=90,
    collocation=linopt.LRrevCollocation(2),
    )

phases = {
    'carbon-switch': [18,54.5,59.5],
    'aerobic-anaerobic': [38,53,71,119],
    'rich-medium': [27,42,49.5,56],
    }


@scripttool.memoize.filecache
def mgoptimize(mg):
    return mg.optimize()

def export_data(task, mg, traj, bm_vec, tf=None, id=None, optgrowth=None, phases=[]):
    if tf is None:
        tf = mg.params['finaltime']
    time = np.linspace(0, tf, num=200)
    x = traj.get_substances(time)
    v = traj.get_fluxes(time)
    bm = np.array([bm_vec.dot(x[i,:]) for i,ti in enumerate(time)])
    Sind = len(mg.ext_species)+mg.gen_species.index('S')
    Straj = bm_vec[Sind] * x[:,Sind]
    Rind = len(mg.ext_species)+mg.gen_species.index('R')
    Rtraj = bm_vec[Rind] * x[:,Rind]
    vgensum = sum( ( v[:, mg.reactions.index(vi)] / mg.catconstants[vi] for vi in mg.vgen) )
    growth = np.diff(bm)
    if optgrowth is not None:
        growth = np.concatenate(([optgrowth * bm[0]], growth))
    else:
        growth = np.concatenate(([growth[0]], growth))
    growth = growth / bm

    file = open(os.path.join(task.get_output_dir(), "data-substrates.csv" if id is None else "data-substrates-%s.csv" % id), "w")
    writer = csv.writer(file, delimiter=" ")
    writer.writerow(["Time",] + ["%s" % i for i in mg.ext_species])
    for i,ti in enumerate(time):
        writer.writerow(["%.5g" % ti] + ["%.5g" % x[i,j] for j,s in enumerate(mg.ext_species)])
    file.close()

    file = open(os.path.join(task.get_output_dir(), "data-biomass.csv" if id is None else "data-biomass-%s.csv" % id), "w")
    writer = csv.writer(file, delimiter=" ")
    writer.writerow(["Time", "Biomass", "Ribosomes", "Structure", "Growthrate"])
    for i,ti in enumerate(time):
        writer.writerow(["%.5g" % ti] + ["%.5g" % data[i] for data in (bm, Rtraj, Straj, growth)])
    file.close()

    file = open(os.path.join(task.get_output_dir(), "data-component-percentages.csv" if id is None else "data-component-percentages-%s.csv" % id), "w")
    writer = csv.writer(file, delimiter=" ")
    writer.writerow(["Time", "Ribosomes", "Structure", "Enzymes"])
    for i,ti in enumerate(time):
        writer.writerow(["%.5g" % ti] + ["%.5g" % data[i] for data in (100 * Rtraj / bm, 100 * Straj / bm, 100 * (1 - (Rtraj + Straj) / bm))])
    file.close()

    file = open(os.path.join(task.get_output_dir(), "data-fluxes.csv" if id is None else "data-fluxes-%s.csv" % id), "w")
    writer = csv.writer(file, delimiter=" ")
    writer.writerow(["Time",] + ["%s" % i for i in mg.vmetab])
    for i,ti in enumerate(time):
        writer.writerow(["%.5g" % ti] + ["%.5g" % (v[i,len(mg.vexc)+j] / bm[i]) for j,vj in enumerate(mg.vmetab)])
    file.close()

    phases = [0,] + phases + [tf,]
    phaselist = [(phases[i], phases[i+1]) for i in range(len(phases)-1)]
    activity_dict = {}
    for i,e in enumerate(mg.gen_species):
        if e.startswith("ER"):
            vind = mg.reactions.index(e[1:])
            for j,p in enumerate(phaselist):
                w = integrate.quad(lambda t: np.abs(traj.get_fluxes(t)[vind] / np.interp(t, time, bm)), p[0], p[1])[0]
                activity_dict[e + "_" + str(j)] = w / (p[1] - p[0])
    maxact = max(activity_dict.values())
    for k,v in activity_dict.items():
        activity_dict[k] = int(np.round(v / maxact * 90))
        activity_dict[k] = activity_dict[k] + (10 if activity_dict[k] > 0 else 0)
            
    db = shelve.open(os.path.join(task.get_output_dir(), "enzyme_activities.db"))
    db['activity'] = activity_dict
    db.close()
                     

def plot_result(task, mg, traj, bm_vec, tf=None, id=None, optgrowth=None):
    if tf is None:
        tf = mg.params['finaltime']
    time = np.linspace(0, tf, num=200)
    x = traj.get_substances(time)
    v = traj.get_fluxes(time)
    fig, ax = task.make_ax(name="plot-substrates" if id is None else "%s-plot-substrates" % id, 
                           xlabel="Time",
                           ylabel="Y",
                           # figtype="small" 
        )
    for i,s in enumerate(mg.ext_species):
        ax.plot(time, x[:,i])
    ax.legend(mg.ext_species)
    fig, ax = task.make_ax(name="plot-biomass" if id is None else "%s-plot-biomass" % id, 
                           xlabel="Time",
                           ylabel="Biomass",
                           # figtype="small" 
        )
    bm = np.array([bm_vec.dot(x[i,:]) for i,ti in enumerate(time)])
    growth = np.diff(bm)
    if optgrowth is not None:
        growth = np.concatenate(([optgrowth * bm[0]], growth))
    else:
        growth = np.concatenate(([growth[0]], growth))
    growth = growth / bm
    ax.semilogy(time, growth, 'k', lw=2, basey=2)
    ax.semilogy(time, bm, 'r', lw=2, basey=2)
    Sind = len(mg.ext_species)+mg.gen_species.index('S')
    Rind = len(mg.ext_species)+mg.gen_species.index('R')
    Straj = bm_vec[Sind] * x[:,Sind]
    Rtraj = bm_vec[Rind] * x[:,Rind]
    ax.semilogy(time, Straj, 'b', basey=2)
    ax.semilogy(time, Rtraj, 'g', basey=2)
    ax.legend(["Biomass", "Structure", "Ribosome"], loc='upper left')
    ax.semilogy(time, mg.structure * bm, 'b--', basey=2)
    if optgrowth is not None:
        optbm = np.clip(bm[0] * np.exp(time * optgrowth), 0, bm[-1])
        ax.semilogy(time, optbm, 'r--', basey=2)

    fig, ax = task.make_ax(name="plot-component-percentages" if id is None else "%s-plot-component-percentages" % id, 
                           xlabel="Time",
                           ylabel="Percentage",
                           # figtype="small" 
        )
    vgensum = sum( ( v[:, mg.reactions.index(vi)] / mg.catconstants[vi] for vi in mg.vgen) )
    ax.plot(time, 100 * Straj / bm, 'b')
    ax.plot(time, 100 * Rtraj / bm, 'g')
    ax.plot(time, 100 * (1 - (Straj + Rtraj) / bm), 'r')
    ax.plot(time, 100 * bm_vec[Rind] * vgensum / bm, 'g--')
    ax.legend(["Structure", "Ribosome", "Enzymes"], loc='upper left')

    if id is not None and len(scen_enzymes[id]) > 0:
        fig, ax = task.make_ax(name="plot-enzymes" if id is None else "%s-plot-enzymes" % id, 
                               xlabel="Time",
                               ylabel="Biomass",
                               # figtype="small" 
            )
        for e in scen_enzymes[id]:
            ax.plot(time, x[:,len(mg.ext_species)+mg.gen_species.index(e)] / bm)
        ax.legend(scen_enzymes[id], loc='upper left')
        # ax.set_ylim(2**-16, 2**-4)
    fig, ax = task.make_ax(name="plot-ribosome" if id is None else "%s-plot-ribosome" % id, 
                           xlabel="Time",
                           ylabel="Fluxes / Ribosome",
                           # figtype="small" 
        )
    for vi in mg.vgen:
        ax.semilogy(time, v[:, mg.reactions.index(vi)], basey=2)
    ax.semilogy(time, v[:, mg.reactions.index("PS")], 'b', lw=2, basey=2)
    vgensum = sum( ( v[:, mg.reactions.index(vi)] / mg.catconstants[vi] for vi in mg.vgen) )
    ax.semilogy(time, vgensum, 'r', lw=2, basey=2)
    ax.semilogy(time, x[:,len(mg.ext_species)+mg.gen_species.index('R')], 'g--', lw=2, basey=2)
    ax.set_ylim(2**-18, 2**-4)

    fig, ax = task.make_ax(name="plot-fluxes" if id is None else "%s-plot-fluxes" % id, 
                           xlabel="Time",
                           ylabel="Relative metabolic fluxes",
                           # figtype="small" 
        )
    fluxes = mg.vexc + mg.vmetab
    for vi in fluxes:
        ax.plot(time, v[:, mg.reactions.index(vi)] / bm)
    ax.legend(fluxes)


class OptMetabo(scripttool.Task):
    """
    Metabogen optimization.
    """
    customize = {
        "discount": None,
        "initial": {},
        "pre_initial": {},
        "rbainit": True,
        "biomass": 0.005,
        "scen_id": None,
        "optparameters": {},
        "o2turnover": False,
        }

    def run(self):
        # initialize through RBA?
        myoptparam = optparameters.copy()
        myoptparam.update(self.optparameters)
        initial = self.initial
        if self.rbainit:
            net, mg, bm_vec = coremetabogen.mg_net(initial=self.pre_initial)
            bm = bm_vec.dot(mg.z0) / mg.alpha if self.biomass is None else self.biomass
            self.printf("Specified biomass: %g" % bm)
            murba, res = mg.rba(tolerance=1e-2, bmvec=bm_vec[-len(mg.gen_species):], biomass=bm, solver='glpk')
            vy, vx, vp, p = res
            bm0 = bm_vec[-len(mg.gen_species):].dot(p)
            self.printf("Maximal growth rate from RBA: %g" % murba)
            self.printf("Initial biomass: %g" % bm0)
            rbastate = dict(zip(mg.gen_species, p))
            self.printf("Optimal state from RBA: %s" % str(rbastate), format=False)
            initial.update(rbastate)
        else:
            murba = None
            rbastate = {}
            
        net, mg, bm_vec = coremetabogen.mg_net(initial=initial, o2turnover=self.o2turnover)
        self.printf(str(net))
        if self.discount is None:
            mg.set_objective(0 * bm_vec, np.zeros(mg.numreact), 0.0, -bm_vec)
        else:
            discountedC = metabogen.DiscountedVector(-bm_vec, self.discount)
            mg.set_objective(discountedC, np.zeros(mg.numreact), 0.0, 0 * bm_vec)

        mg.set_parameters(**myoptparam)
            
        tf, obj, traj = mgoptimize(mg)
        self.printf("Optimal objective function value: %g" % obj)
        self.printf("Initial rates:")
        v0 = traj.get_fluxes(0)
        for r, vi in zip(mg.reactions, v0):
            self.printf("%s = %g" % (r, vi))
        plot_result(self, mg, traj, bm_vec, id=self.scen_id, optgrowth=murba)
        export_data(self, mg, traj, bm_vec, id=self.scen_id, phases=phases[self.scen_id])
        
class OptMetaboScenarios(scripttool.Task):
    """
    Metabogen optimization.
    """
    customize = {"alpha": 10,
                 "discount": None,
                 "biomass": 1e-4}

    def run(self):
        # initialize through RBA
        net, mg, bm_vec = coremetabogen.mg_net()
        bm = bm_vec.dot(mg.z0) / mg.alpha if self.biomass is None else self.biomass
        self.printf("Specified biomass: %g" % bm)
        mu, res = mg.rba(tolerance=1e-4, bmvec=bm_vec[-len(mg.gen_species):], solver='glpk')
        vy, vx, vp, p = res
        bm0 = bm_vec[-len(mg.gen_species):].dot(p * bm)
        self.printf("Maximal growth rate from RBA: %g" % mu)
        self.printf("Initial biomass: %g" % bm0)
        rbastate = dict(zip(mg.gen_species, p * bm))
        self.printf("Optimal state from RBA: %s" % str(rbastate))

        for s in scenarios:
            self.printf("Scenario %s initial condition:" % s)
            for i in scenarios[s]:
                self.printf("%s = %g" % (i, scenarios[s][i]), indent=1)
            scenarios[s].update(rbastate)
            net, mg, bm_vec = coremetabogen.mg_net(initial=scenarios[s])
            if self.discount is None:
                mg.set_objective(0 * bm_vec, np.zeros(mg.numreact), 0.0, -bm_vec)
            else:
                discountedC = metabogen.DiscountedVector(-bm_vec, self.discount)
                mg.set_objective(discountedC, np.zeros(mg.numreact), 0.0, 0 * bm_vec)

            mg.set_parameters(finaltime=40., timesteps=20, collocation=linopt.LRrevCollocation(3))
            mg.set_parameters(solver='glpk')

            tf, obj, traj = mgoptimize(mg)
            self.printf("Optimal objective function value: %g" % obj)
            plot_result(self, mg, traj, bm_vec, id=s)
        
# creation of my experiments
# scripttool.register_task((taskoption=["custom"]), ident="task_custom")
t = OptMetabo()
t.optparameters['solver'] = None
scripttool.register_task(t, ident="optcoremetabogen-terminal")
scripttool.register_task(OptMetabo(discount=0.1), ident="optcoremetabogen-discount")
t = OptMetabo(discount=0.1, initial=dict(Carb1=2, Carb2=30, O2ext=50),
              pre_initial=dict(Carb1=2, Carb2=30, O2ext=50),
              scen_id="carbon-switch")
t.optparameters['finaltime'] = 90.
scripttool.register_task(t, ident="optcoremetabogen-carbon-switch")
t = OptMetabo(discount=0.1, initial=dict(Carb1=50, Carb2=0, O2ext=2.5),
              pre_initial=dict(Carb1=50, Carb2=0, O2ext=2.5),
              scen_id="aerobic-anaerobic", o2turnover=(1, 0.4))
t.optparameters['finaltime'] = 140.
scripttool.register_task(t, ident="optcoremetabogen-anaerobic-switch")
t = OptMetabo(discount=0.1, initial=dict(Carb1=50, Carb2=0, O2ext=50, Hext=5, Fext=5),
              pre_initial=dict(Carb1=50, Carb2=0, O2ext=50, Hext=0, Fext=0),
              scen_id="rich-medium")
t.optparameters['finaltime'] = 75.
scripttool.register_task(t, ident="optcoremetabogen-rich-medium")
# scripttool.register_task(OptMetaboScenarios(), ident="optcoremetabogen-scenarios")
# scripttool.register_task(OptMetaboScenarios(discount=0.3), ident="optcoremetabogen-scenarios-discount")
