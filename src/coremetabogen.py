"""
Module coremetabogen.py: set up core metabolic-genetic network
"""

import numpy as np

import brn

import metabogen

def mg_net(discount=None, initial={}, o2turnover=None):
    net = brn.fromSBML("models/core_metabolism.xml")
    if len(initial):
        net.set(initial)
    cmgnet_ext_species = ["Carb1", "Carb2", "O2ext", "Dext", "Eext", "Fext", "Hext"]
    cmgnet_metab_species = ["A", "B", "C", "D", "E", "F", "G", "H", "ATP", "NADH", "O2"]
    cmgnet_gen_species = ["ETc1", "ETc2", "ETf", "ETh", "ER1", "ER2", "ER3", "ER4", "ER5", "ER6", "ER7", "ER8", "ERres", "R", "S"]
    cmgnet_enzymes = dict.fromkeys(["ETc1", "ETc2", "ETf", "ETh", "ER1", "ER2", "ER3", "ER4", "ER5", "ER6", "ER7", "ER8", "ERres", "R", "S"])
    for e in cmgnet_enzymes:
	cmgnet_enzymes[e] = []
        if e != "R" and e != "S":
            cmgnet_enzymes[e].append(e[1:])
    for r in net.reactions:
        if r.startswith("D"):
            cmgnet_enzymes['S'].append(r)
        if r.startswith("P"):
            cmgnet_enzymes['R'].append(r)
    catconstants = {}
    if o2turnover is None:
        net.set({"vO2":0.0, "muO2":0.0})
    else:
        if np.iterable(o2turnover):
            net.set({"vO2":o2turnover[0], "muO2":o2turnover[1]})
    ext_react = ["vO"]
    for r in net.reactions:
        if r not in ext_react:
            catconstants[r] = float(net.values[r])
    mg = metabogen.Metabogen(net, cmgnet_ext_species, cmgnet_metab_species, cmgnet_gen_species,
                             enzymes=cmgnet_enzymes, catconstants=catconstants, alpha=100,
                             ext_react=ext_react)
    # check on biomass composition
    Sbm = mg.Spp.dot(mg.Sxp.T)
    bm_vec = np.concatenate( (np.zeros(len(cmgnet_ext_species)), np.ones(len(cmgnet_gen_species))) )
    for i,s in enumerate(mg.gen_species):
        pass
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ETc1")] = 4
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ETc2")] = 15
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ETf")] = 4
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ETh")] = 4
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER1")] = 5
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER2")] = 5
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER3")] = 20
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER4")] = 5
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER5")] = 5
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER6")] = 10
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER7")] = 10
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ER8")] = 40
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("ERres")] = 5
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("R")] = 60
    bm_vec[len(cmgnet_ext_species)+cmgnet_gen_species.index("S")] = 7.5
    mg.set_objective(0*bm_vec, np.zeros(len(net.reactions)), 0, -bm_vec)
    vmin = np.zeros(len(net.reactions))
    for rrev in ["R2", "R6", "R7", "R8", "Dd", "De"]:
        vmin[net.reactions.index(rrev)] = -np.inf
    mg.set_reaction_bounds(vmin=vmin)
    mg.structure = 0.35
    struc_constr = mg.structure * bm_vec
    Sindex = len(cmgnet_ext_species)+cmgnet_gen_species.index("S")
    struc_constr[Sindex] -= bm_vec[Sindex]
    mg.add_constraint(np.atleast_2d(struc_constr), np.zeros((1,mg.numreact)), np.array([0.]))
    return net, mg, bm_vec


