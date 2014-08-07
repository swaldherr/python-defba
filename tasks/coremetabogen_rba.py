"""
RBA of the core metabolic genetic network
"""
# Time-stamp: <Last change 2013-04-25 15:24:29 by Steffen Waldherr>

import numpy as np

import scripttool

from src import coremetabogen

class CoremetabogenRBA(scripttool.Task):
    """
    Do RBA of the core metobolic-genetic network.
    """
    customize = {
        "initial": {}
        }

    def run(self):
        net, mg, bmvec = coremetabogen.mg_net(initial=self.initial)
        bm = bmvec.dot(mg.z0) / mg.alpha
        self.printf("Specified biomass = %g" % bm)
        mu, res = mg.rba(tolerance=1e-3, bmvec=bmvec[-len(mg.gen_species):], biomass=bm, solver='glpk')
        vy, vx, vp, p = res
        self.printf("muopt = %s" % str(mu))
        self.printf("vy = %s" % str(vy))
        self.printf("vx = %s" % str(vx))
        self.printf("vp = %s" % str(vp))
        self.printf("p = %s" % str(p))
        self.printf("Result biomass = %g" % bmvec[-len(mg.gen_species):].dot(p))

class CoremetabogenRBAMu(scripttool.Task):
    """
    Do RBA of the core metobolic-genetic network for specific growth rates.
    """
    customize = {"mu":[0.05, 0.1, 0.15, 0.2, 0.25]}

    def run(self):
        net, mg, bmvec = coremetabogen.mg_net()
        bm = bmvec.dot(mg.z0) / mg.alpha
        self.printf("Specified biomass = %g" % bm)
        for mui in self.mu:
            self.printf("RBA for mu = %g:" % mui)
            r, res = mg.rba_mu(mu=mui, bmvec=bmvec[-len(mg.gen_species):], biomass=bm)
            if r:
                vy, vx, vp, p = res
                self.printf("vy = %s" % str(vy), indent=1)
                self.printf("vx = %s" % str(vx), indent=1)
                self.printf("vp = %s" % str(vp), indent=1)
                self.printf("p = %s" % str(p), indent=1)
                self.printf("Result biomass = %g" % bmvec[-len(mg.gen_species):].dot(p), indent=1)
            else:
                self.printf("Infeasible growth rate.", indent=1)
        
        # fig, ax = self.make_ax(name="figurename",
        #                        xlabel="x",
        #                        ylabel="y",
        #                        title="")

# creation of my experiments
# scripttool.register_task((taskoption=["custom"]), ident="task_custom")
scripttool.register_task(CoremetabogenRBAMu(), ident="coremetabogen-rba-mu")
scripttool.register_task(CoremetabogenRBA(), ident="coremetabogen-rba")
scripttool.register_task(CoremetabogenRBA(initial=dict(Carb1=0, Carb2=10, O2ext=10)), ident="coremetabogen-rba-carb2")
