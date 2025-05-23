#bond from charmm36 topology

bond_dict = {\
    'ALA': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "), #+N does not work. - minor problem.
             (" CA "," CB "),
           ],
    'ARG': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD "),
             (" CD "," NE "),
             (" NE "," CZ "),
             (" CZ "," NH1"),
             (" CZ "," NH2"),
           ],           
    'ASN': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," OD1"),
             (" CG "," ND2"),
           ],  
    'ASP': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," OD1"),
             (" CG "," OD2"),
           ],  
    'CYS': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," SG "),
           ],
    'GLN': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD "),
             (" CD "," OE1"),
             (" CD "," NE2"),
           ],    
    'GLU': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD "),
             (" CD "," OE1"),
             (" CD "," OE2"),
           ],  
    'GLY': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
           ],
    'HIS': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," ND1"),
             (" ND1"," CE1"),
             (" CE1"," NE2"),
             (" NE2"," CD2"),
             (" CD2"," CG "),
           ],  
    'ILE': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG1"),
             (" CG1"," CD1"),
             (" CB "," CG2"),
           ],
    'LEU': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD1"),
             (" CG "," CD2"),
           ],
    'LYS': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD "),
             (" CD "," CE "),
             (" CE "," NZ "),
           ],
    'MET': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," SD "),
             (" SD "," CE "),
           ],
    'PHE': [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD1"),
             (" CD1"," CE1"),
             (" CE1"," CZ "),
             (" CG "," CD2"),
             (" CD2"," CE2"),
             (" CE2"," CZ "),
           ],
    "PRO": [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD "),
             (" CD "," N  "),
           ],
    "SER": [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," OG "),
           ],
    "THR": [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," OG1"),
             (" CB "," CG2"),
           ],
    "TRP": [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD1"),
             (" CD1"," NE1"),
             (" NE1"," CE2"),
             (" CE2"," CD2"),
             (" CG "," CD2"),
             (" CD2"," CE3"),
             (" CE3"," CZ3"),
             (" CZ3"," CH2"),
             (" CH2"," CZ2"),
             (" CE2"," CZ2"),
           ],
    "TYR": [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG "),
             (" CG "," CD1"),
             (" CD1"," CE1"),
             (" CE1"," CZ "),
             (" CG "," CD2"),
             (" CD2"," CE2"),
             (" CE2"," CZ "),
             (" CZ "," OH "),
           ],
    "VAL": [ (" N  "," CA "),
             (" CA "," C  "),
             (" C  "," O  "),
             (" C  ","+N  "),
             (" CA "," CB "),
             (" CB "," CG1"),
             (" CB "," CG2"),
           ]
}
#                     01234   01234 (base vec)
polar_vec_dict = {   
    'ALA': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  ']
           },
    'ARG': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " NE ":[ ' CD ',' CZ '],
             " NH1":[ ' CZ '],
             " NH2":[ ' CZ '],
           },
    'ASN': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OD1":[ ' CG '],
             " ND2":[ ' CG '],
           },
    'ASP': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OD1":[ ' CG '],
             " OD2":[ ' CG '],
           },
    'CYS': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
    'GLN': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OE1":[ ' CD '],
             " NE2":[ ' CD '],
           },
    'GLU': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OE1":[ ' CD '],
             " OE2":[ ' CD '],
           },
    'GLY': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
    'HIS': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " ND1":[ ' CG ', ' CE1'],
             " NE2":[ ' CE1', ' CD2'],             
           },
    'ILE': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
    'LEU': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
    'LYS': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " NZ ":[ ' CE '],
           },
    'MET': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
    'PHE': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
    'PRO': { " O  ":[ ' C  '],
           },
    'SER': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OG ":[ ' CB '],
           },
    'THR': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OG1":[ ' CB '],
           },
    'TRP': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " NE1":[ ' CD1', ' CE2'],
           },
    'TYR': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
             " OH ":[ ' CZ '],
           },
    'VAL': { " N  ":['- C  ', ' CA '],
             " O  ":[ ' C  '],
           },
}

#another vector for building local coordinte for polar atom 
#local coordinate system:
#axis 0: normalized vector of (polar_vec)
#axis 1: normalized vector of (aux_vec - dot(aux_vec, axis 0)*aux_vec)
#axis 2: cross(axis 0, axis 1)

aux_vec_dict_old = {   
    'ALA': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '], to ensure axis0 != axis1, even for terminals 
             " O  ":[ ' CA ']  #axis 0: [ ' C  ']
           },
    'ARG': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " NE ":[ ' CD '], #axis 0: [ ' CD ',' CZ '],
             " NH1":[ ' NE '], #axis 0: [ ' CZ ']
             " NH2":[ ' NE '], #axis 0: [ ' CZ ']
           },
    'ASN': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OD1":[ ' CB '], #axis 0: [ ' CG '],
             " ND2":[ ' CB '], #axis 0: [ ' CG '],
           },
    'ASP': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OD1":[ ' CB '], #axis 0: [ ' CG '],
             " OD2":[ ' CB '], #axis 0: [ ' CG '],
           },
    'CYS': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'GLN': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OE1":[ ' CG '], #axis 0: [ ' CD '],
             " NE2":[ ' CG '], #axis 0: [ ' CD '],
           },
    'GLU': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OE1":[ ' CG '], #axis 0: [ ' CD '],
             " OE2":[ ' CG '], #axis 0: [ ' CD '],
           },
    'GLY': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'HIS': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " ND1":[ ' CG '], #axis 0:[ ' CG ', ' CE1']
             " NE2":[ ' CD2'], #axis 0:[ ' CE1', ' CD2']         
           },
    'ILE': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'LEU': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'LYS': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " NZ ":[ ' CD '], #axis 0: [ ' CE ']
           },
    'MET': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'PHE': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'PRO': { " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'SER': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OG ":[ ' CA '], #axis 0: [ ' CB ']
           },
    'THR': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OG1":[ ' CA '], #axis 0: [ ' CB ']
           },
    'TRP': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " NE1":[ ' CD1'], #axis 0: [ ' CD1', ' CE2']
           },
    'TYR': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OH ":[ ' CA '], #axis 0: [ ' CZ '], since CDE1/CDE2 is equivalent, CG/CB is parallel to CZ
           },
    'VAL': { " N  ":[ ' C  '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
}

aux_vec_dict = {   
    'ALA': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '], C->CA. 
             " O  ":[ ' CA ']  #axis 0: [ ' C  ']
           },
    'ARG': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " NE ":[ ' CD '], #axis 0: [ ' CD ',' CZ '],
             " NH1":[ ' NE '], #axis 0: [ ' CZ ']
             " NH2":[ ' NE '], #axis 0: [ ' CZ ']
           },
    'ASN': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OD1":[ ' CB '], #axis 0: [ ' CG '],
             " ND2":[ ' CB '], #axis 0: [ ' CG '],
           },
    'ASP': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OD1":[ ' CB '], #axis 0: [ ' CG '],
             " OD2":[ ' CB '], #axis 0: [ ' CG '],
           },
    'CYS': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'GLN': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OE1":[ ' CG '], #axis 0: [ ' CD '],
             " NE2":[ ' CG '], #axis 0: [ ' CD '],
           },
    'GLU': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OE1":[ ' CG '], #axis 0: [ ' CD '],
             " OE2":[ ' CG '], #axis 0: [ ' CD '],
           },
    'GLY': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'HIS': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " ND1":[ ' CG '], #axis 0:[ ' CG ', ' CE1']
             " NE2":[ ' CD2'], #axis 0:[ ' CE1', ' CD2']         
           },
    'ILE': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'LEU': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'LYS': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " NZ ":[ ' CD '], #axis 0: [ ' CE ']
           },
    'MET': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'PHE': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'PRO': { " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
    'SER': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OG ":[ ' CA '], #axis 0: [ ' CB ']
           },
    'THR': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OG1":[ ' CA '], #axis 0: [ ' CB ']
           },
    'TRP': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " NE1":[ ' CD1'], #axis 0: [ ' CD1', ' CE2']
           },
    'TYR': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
             " OH ":[ ' CE1'], #axis 0: [ ' CZ '], CA ->CE1
           },
    'VAL': { " N  ":[ ' CA '], #axis 0: ['- C  ', ' CA '],
             " O  ":[ ' CA '], #axis 0: [ ' C  ']
           },
}
charge_dict = {
    'ALA': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.27,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'ARG': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.18,
            ' CD ':0.20,
            ' NE ':-0.70,
            ' CZ ':0.64,
            ' NH1':-0.80,
            ' NH2':-0.80,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'ASN': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':0.55,
            ' OD1':-0.55,
            ' ND2':-0.62,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'ASP': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.28,
            ' CG ':0.62,
            ' OD1':-0.76,
            ' OD2':-0.76,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'CYS': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.11,
            ' SG ':-0.23,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'GLN': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.18,
            ' CD ':0.55,
            ' OE1':-0.55,
            ' NE2':-0.62,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'GLU': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.28,
            ' CD ':0.62,
            ' OE1':-0.76,
            ' OE2':-0.76,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'GLY': {' N  ':-0.47,
            ' CA ':-0.02,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'HIS': {' N  ':-0.47, #based on HSD (neutral HIS)
            ' CA ':0.07,
            ' CB ':-0.09,
            ' ND1':-0.36,
            ' CG ':-0.05,
            ' CE1':0.25,
            ' NE2':-0.70,
            ' CD2':0.22,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'ILE': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.09,
            ' CG2':-0.27,
            ' CG1':-0.18,
            ' CD1':-0.27,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'LEU': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.09,
            ' CD1':-0.27,
            ' CD2':-0.27,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'LYS': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.18,
            ' CD ':-0.18,
            ' CE ':0.21,
            ' NZ ':-0.30,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'MET': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.14,
            ' SD ':-0.09,
            ' CE ':-0.22,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'PHE': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':0.00,
            ' CD1':-0.115,
            ' CE1':-0.115,
            ' CZ ':-0.115,
            ' CD2':-0.115,
            ' CE2':-0.115,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'PRO': {' N  ':-0.29,
            ' CD ':0.00,
            ' CA ':0.02,
            ' CB ':-0.18,
            ' CG ':-0.18,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'SER': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':0.05,
            ' OG ':-0.66,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'THR': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':0.14,
            ' OG1':-0.66,
            ' CG2':-0.27,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'TRP': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':-0.03,
            ' CD1':-0.15,
            ' NE1':-0.51,
            ' CE2':0.24,
            ' CD2':0.11,
            ' CE3':-0.25,
            ' CZ3':-0.20,
            ' CZ2':-0.27,
            ' CH2':-0.14,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'TYR': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.18,
            ' CG ':0.00,
            ' CD1':-0.115,
            ' CE1':-0.115,
            ' CZ ':0.11,
            ' OH ':-0.54,
            ' CD2':-0.115,
            ' CE2':-0.115,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
    'VAL': {' N  ':-0.47,
            ' CA ':0.07,
            ' CB ':-0.09,
            ' CG2':-0.27,
            ' CG1':-0.27,
            ' C  ':0.51,
            ' O  ':-0.51,
    },
}

hyb_dict = { #for discriminating sp2 O and sp3 O. marked as sp2: 0, sp3: 1
    'ALA': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'ARG': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' CD ':1,
            ' NE ':0,
            ' CZ ':0,
            ' NH1':0,
            ' NH2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'ASN': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':0,
            ' OD1':0,
            ' ND2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'ASP': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':0,
            ' OD1':0,
            ' OD2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'CYS': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' SG ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'GLN': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' CD ':0,
            ' OE1':0,
            ' NE2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'GLU': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' CD ':0,
            ' OE1':0,
            ' OE2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'GLY': {' N  ':0,
            ' CA ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'HIS': {' N  ':0, #based on HSD (neutral HIS)
            ' CA ':1,
            ' CB ':1,
            ' ND1':0,
            ' CG ':0,
            ' CE1':0,
            ' NE2':0,
            ' CD2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'ILE': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG2':1,
            ' CG1':1,
            ' CD1':1,
            ' C  ':0,
            ' O  ':0,
    },
    'LEU': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' CD1':1,
            ' CD2':1,
            ' C  ':0,
            ' O  ':0,
    },
    'LYS': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' CD ':1,
            ' CE ':1,
            ' NZ ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'MET': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' SD ':1,
            ' CE ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'PHE': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':0,
            ' CD1':0,
            ' CE1':0,
            ' CZ ':0,
            ' CD2':0,
            ' CE2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'PRO': {' N  ':0,
            ' CD ':1,
            ' CA ':1,
            ' CB ':1,
            ' CG ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'SER': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' OG ':1,
            ' C  ':0,
            ' O  ':0,
    },
    'THR': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' OG1':1,
            ' CG2':1,
            ' C  ':0,
            ' O  ':0,
    },
    'TRP': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':0,
            ' CD1':0,
            ' NE1':0,
            ' CE2':0,
            ' CD2':0,
            ' CE3':0,
            ' CZ3':0,
            ' CZ2':0,
            ' CH2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'TYR': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG ':0,
            ' CD1':0,
            ' CE1':0,
            ' CZ ':0,
            ' OH ':1,
            ' CD2':0,
            ' CE2':0,
            ' C  ':0,
            ' O  ':0,
    },
    'VAL': {' N  ':0,
            ' CA ':1,
            ' CB ':1,
            ' CG2':1,
            ' CG1':1,
            ' C  ':0,
            ' O  ':0,
    },
}