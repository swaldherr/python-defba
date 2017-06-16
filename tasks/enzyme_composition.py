"""
Compute enzyme amino acid compositions
"""
# Time-stamp: <Last change 2017-06-16 14:32:59 by Steffen Waldherr>

import numpy as np
import os.path
import csv
import warnings

import scripttool
from Bio import SeqIO
from Bio import Entrez

from src import biodata

class EnzymeComposition(scripttool.Task):
    """
    
    """
    customize = {"enzymefile":None,
                 "download":True,
                 "gbfiledir":os.path.join("data", "ncbi-files"),
                 "entrezemail": None
                 }

    def run(self):
        if self.entrezemail is None:
            Entrez.email = raw_input("Enter email address to be used with Entrez: ")
        else:
            Entrez.email = self.entrezemail
        with open(self.enzymefile, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            headers = reader.next()
            data = [[i for i in row] for row in reader]
        self.enzymes = dict()
        for entry in data:
            composition_num = eval(entry[3])
            composition = dict(((str(i), j) for i,j in composition_num.items()))
            if sum(composition.values()) > 1:
                e = biodata.Protein(name=entry[0], iscomplex=True)
                e.set_peptide_composition(**composition)
                for k in composition:
                    if k not in self.enzymes:
                        ek = biodata.Protein(name=k, gid=k)
                        self.enzymes[k] = ek
            else:
                e = biodata.Protein(name=entry[0], iscomplex=False, gid=composition.keys()[0])
            self.enzymes[entry[0]] = e
            e.url = entry[1]
        if self.download:
            for e in self.enzymes:
                if self.enzymes[e].gid is not None:
                    try:
                        self.enzymes[e].download_sequence()
                        SeqIO.write(self.enzymes[e].seq_record, os.path.join(self.get_output_dir(), e + ".gb"), format="gb")
                    except Exception, err:
                        warnings.warn("For enzyme " + e + ": " + str(err))
        else:
            for e in self.enzymes:
                if self.enzymes[e].gid is not None:
                    self.enzymes[e].load_sequence(os.path.join(self.gbfiledir, e + ".gb"))
        for e in self.enzymes.values():
            e.compute_composition(self.enzymes)
            e.compute_weight(self.enzymes)
        median_weight = np.median([e.get_weight(self.enzymes) for e in self.enzymes.values() if e.name != e.gid])
        for e in self.enzymes.values():
            e.estimate_kcat(median_weight=median_weight)
        with open(os.path.join(self.get_output_dir(), "enzymes_with_composition.csv"), "w") as resfile:
            reswriter = csv.writer(resfile, delimiter=',')
            headers.append("Length")
            aa_list = biodata.amino_acids.values()
            headers.extend(aa_list)
            headers.append("Weight")
            headers.append("est. kcat [1/min]")
            reswriter.writerow(headers)
            names = [i[0] for i in data]
            for e in self.enzymes.values():
                if e.gid != e.name:
                    row = data[names.index(e.name)]
                    row.append(sum(( count for count in e.composition.values() )))
                    for aa in aa_list:
                        row.append(e.composition[aa])
                    row.append(e.weight)
                    row.append(e.kcat)
                    reswriter.writerow(row)
        
# creation of my experiments
scripttool.register_task(EnzymeComposition(enzymefile="models/e-coli-core-carbon-enzymes.csv"), ident="ecoli-core-enzyme-composition")
scripttool.register_task(EnzymeComposition(enzymefile="models/e-coli-core-carbon-enzymes.csv", download=False), ident="ecoli-core-enzyme-composition-preloaded")
