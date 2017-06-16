"""
Module biodata.py: tools to obtain data for metabolic-genetic networks
"""

import numpy as np
import os
import csv
import warnings
from collections import defaultdict, OrderedDict

import json

import Bio
from Bio import Seq, SeqIO
from Bio import SeqUtils
from Bio import Entrez
from Bio.Alphabet import IUPAC
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import periodictable

amino_acids = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    }

rna_nucleotides = { # transcription requires the nTP form!
    "A": "ATP",
    "G": "GTP",
    "U": "UTP",
    "C": "CTP",
    }

# from http://home.cc.umanitoba.ca/~alvare/tools/gmwcalc.py
aaweights = {
	'A' : 71.09,  # alanine
	'R' : 156.19, # arginine
	'D' : 114.11, # aspartic acid
	'N' : 115.09, # asparagine
	'C' : 103.15, # cysteine
	'E' : 129.12, # glutamic acid
	'Q' : 128.14, # glutamine
	'G' : 57.05,  # glycine
	'H' : 137.14, # histidine
	'I' : 113.16, # isoleucine
	'L' : 113.16, # leucine
	'K' : 128.17, # lysine
	'M' : 131.19, # methionine
	'F' : 147.18, # phenylalanine
	'P' : 97.12,  # proline
	'S' : 87.08,  # serine
	'T' : 101.11, # threonine
	'W' : 186.12, # tryptophan
	'Y' : 163.18, # tyrosine
	'V' : 99.14   # valine
}

class Protein(object):
    def __init__(self, name, iscomplex=False, gid=None):
        self.name = name
        self.iscomplex = iscomplex
        self.gid = gid

    def set_peptide_composition(self, **kwarg):
        """
        Set this proteins peptide composition. kwarg's are of the form name=num, where name is a protein name
        and num is the number of components of name in this protein (complex).
        """
        self.num_components = sum(kwarg.values())
        self.iscomplex = (self.num_components > 1)
        self.components = kwarg

    def download_sequence(self, gid=None):
        """
        Get this protein's gene sequence from NCBI.
        """
        if gid is not None:
            self.gid = gid
        handle = Entrez.efetch(db="protein", id=self.gid, rettype='gb')
        self.seq_record = SeqIO.read(handle, format='gb')

    def load_sequence(self, file, gid=None):
        """
        Get this protein's gene sequence from a gb file record
        """
        if gid is not None:
            self.gid = gid
        with open(file, "rU") as handle:
            self.seq_record = SeqIO.parse(handle, format='gb').next()

    def compute_composition(self, enzyme_dict={}):
        """
        Compute composition of amino acids for this protein.
        enzyme_dict should be a dict of other proteins to determine composition of protein complexes
        """
        composition = defaultdict(lambda: 0)
        if not self.iscomplex:
            try:
                seq = self.seq_record.seq
            except AttributeError:
                self.download_sequence(self)
                seq = self.seq_record.seq
            for aa in str(seq):
                composition[amino_acids[aa]] += 1
        else:
            for name, coeff in self.components.items():
                subcomposition = enzyme_dict[name].get_composition()
                for c,n in subcomposition.items():
                    composition[c] += coeff * n
        self.composition = composition
        return composition

    def get_composition(self, enzyme_dict={}):
        try:
            return self.composition
        except AttributeError:
            self.compute_composition(enzyme_dict)
            return self.composition

    def compute_weight(self, enzyme_dict={}):
        """
        Get molecular weight of this protein.
        """
        if not self.iscomplex:
            try:
                weight = ProteinAnalysis(str(self.seq_record.seq)).molecular_weight()
            except AttributeError:
                self.download_sequence()
                weight = ProteinAnalysis(str(self.seq_record.seq)).molecular_weight()
        else:
            weight = sum(( coefficient * enzyme_dict[name].get_weight(enzyme_dict)
                           for name, coefficient in self.components.items()))
        self.weight = weight
        return weight

    def get_weight(self, enzyme_dict={}):
        try:
            return self.weight
        except AttributeError:
            self.compute_weight(enzyme_dict)
            return self.weight

    def estimate_kcat(self, median_kcat=3900., median_weight=1.0):
        """
        median_kcat = 3900 1/min from Brien et al., MSB 2013
        """
        w = self.get_weight()
        sasa = w ** 0.75 # Brien et al., MSB 2013; Miller et al., Nature 1987
        self.kcat = sasa / (median_weight ** 0.75) * median_kcat
        return self.kcat
    
        

class CellDatabase(dict):
    def __init__(self, folder=None, *args, **kwargs):
        super(CellDatabase, self).__init__(self, *args, **kwargs)
        self.load_files(folder)
    
    def load_files(self, folder):
        """
        Load json files from folder
        """
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(".json"):
                    try:
                        dat = json.load(open(os.path.join(root, f)))["data"]
                    except Exception as err:
                        dat = []
                        warnings.warn("Error {} occured while trying to load {} as json file.".format(err, f))
                    for d in dat:
                        self[d["wid"]] = d
        self.wid_by_model = defaultdict(lambda: [])
        for wid in self:
            self.wid_by_model[self[wid]['model']].append(wid)                

    def get_weight(self, wid, gene_as_rna=False):
        """
        Get molecular weight of a molecule in the database.
        """
        if self[wid]['model'] == "Metabolite":
            return periodictable.formula(self[wid]['empirical_formula']).mass
        elif self[wid]['model'] == "ProteinMonomer":
            return ProteinAnalysis(str(self.get_sequence(wid))).molecular_weight()
        elif self[wid]['model'] == "ProteinComplex":
            weight = sum(( -float(c['coefficient']) * self.get_weight(c['molecule'], gene_as_rna=True)
                           for c in self[wid]['biosynthesis'] if float(c['coefficient']) < 0 ))
            for c in self[wid]['biosynthesis']:
                if float(c['coefficient']) > 0 and c['molecule'] != wid:
                    weight -= self.get_weight(c['molecule'], gene_as_rna=True)
            return weight
        elif self[wid]['model'] == "Gene":
            if gene_as_rna:
                return SeqUtils.molecular_weight(self.get_sequence(wid).transcribe())
            else:
                return SeqUtils.molecular_weight(self.get_sequence(wid))
        else:
            raise ValueError("Cannot compute weight for element {} of type {}".format(wid, self[wid]['model']))

    def get_sequence(self, wid, type=None, gene_as_rna=False):
        """
        Get sequence (NT or AA) of cellular object, depending on type (Gene, ProteinMonomer, Rna).
        """
        if type is None:
            type = self[wid]['model']
        if gene_as_rna and type == "Gene":
            type = "Rna"
        if type == "ProteinMonomer":
            gene = self[self[wid]['gene']]
        else:
            gene = self[wid]
        seq_ind = int(gene['coordinate']) - 1
        seq_len = int(gene['length'])
        if gene['direction'].lower() == 'forward':
            gene_seq = Seq.Seq(self[gene['chromosome']]['sequence'][seq_ind:(seq_ind + seq_len)], IUPAC.unambiguous_dna)
        else:
            gene_seq = Seq.Seq(self[gene['chromosome']]['sequence'][seq_ind:(seq_ind + seq_len)], IUPAC.unambiguous_dna).reverse_complement()
        if type == "Rna":
            return gene_seq.transcribe()
        elif type == "ProteinMonomer":
            # M. genitalium should use translation table 4 according to http://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG11
            try:
                return gene_seq.translate(table=4, to_stop=True, cds=True) 
            except Bio.Data.CodonTable.TranslationError: # May be caused by sequence errors in the wholecellkb
                warnings.warn("Gene {} may contain a sequence error, using approximate translation.".format(wid))
                return gene_seq.translate(table=4, to_stop=True, cds=False) 
        else:
            return gene_seq

    def get_protein_sequence(self, protid):
        """
        Get AA sequence for protein monomer with protid.
        """
        if self[protid]['model'] != "ProteinMonomer":
            raise ValueError("{} is a {}, not a protein monomer.".format(protid, self[protid]['model']))
        return self.get_sequence(protid)

    def get_composition(self, wid):
        """
        Compute composition of basic metabolites for biomass molecules (proteins or rna).
        """
        composition = defaultdict(lambda: 0)
        if self[wid]['model'] == "Metabolite":
            composition[wid] = 1
        elif self[wid]['model'] == "Gene": # compute gene sequence and return corresponding nucleotides
            seq = self.get_sequence(wid, gene_as_rna=True)
            for na in str(seq):
                composition[rna_nucleotides[na]] += 1
        elif self[wid]['model'] == "ProteinMonomer":
            seq = self.get_sequence(wid)
            for aa in str(seq):
                composition[amino_acids[aa]] += 1
        elif self[wid]['model'] == "ProteinComplex":
            for c in self[wid]['biosynthesis']:
                coeff = int(float(c['coefficient']))
                if coeff < 0:
                    subcomposition = self.get_composition(c['molecule'])
                    for c,n in subcomposition.items():
                        composition[c] += (-coeff) * n
                elif c['molecule'] != wid: # all other biosynthesis components with positive coefficient seem to be basic metabolites!
                    assert(self[c['molecule']]['model'] == "Metabolite")
                    composition[str(c['molecule'])] -= coeff
        return composition

def metab_basename(name):
    return "_".join(name.split("_")[:-1])

def ObjFactory(object):
    """
    Helper class for making pickleable defaultdicts.
    """
    def __init__(self, obj):
        self.obj = obj
    def __call__(self):
        return self.obj

