pdbbind_crude.txt: 397 structures from PDBBind refined set for performance comparison. Structures were culled with the following criteria: 
resolution ≤2.0 Å, <30% sequence identity, ligand Tanimoto <50%, and 5-20% crystallographic water molecules. 

excluded.txt:  27 excluded structures from the actual performance comparison from either of the issues. 
1) The structure does not have crystallographic water near ligand and cannot be used for current performance comparison metric (23 structures)   
2) 3D-RISM predicted 0 ligand-neighboring water positions  (4 structures)

pdbbind_clean.txt: total 370 structures, originated from "test_pdbbind_crude.txt" and excluded 27 structured from "excluded.txt". this set was used for the actual performance comparison.