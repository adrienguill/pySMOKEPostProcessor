from pySMOKEPostProcessor import elements_balance

# -------------------------------------------------------------------------------------
#  - - - Elements balance analysis - - -
# For an OpenSMOKE temperature sweep (one folder per Case), this returns one
# subplot per element, showing which species carries that element, and how
# much of it, across all Cases.
#
# All plotting utilities are built inside the post-processor function itself;
# calling elements_balance(...) does the parsing and shows the plots directly.
#
# How to use:
#  kinetic_folder   -- folder containing the mech (kinetics.xml is read), read the
#                      elements composition per species
#  results_folders  -- folder containing the Cases (Output.xml are read)
#  elements_list    -- elements symbols to plot, in a list of string as:["C", "O", "H"]
#                      any cases is accepted, if not present in any species is skipped
#  threshold        -- *optional* puts a threshold at which a species is individually
#                      shown, under it is put in the "Others", base value is 5%
# -------------------------------------------------------------------------------------


kineticFolder = "/home/aguillonneau/Creck/OMEsCK"
resultsFolder= "/home/aguillonneau/opensmoke/Wang/OME3/1/out"


elements_balance(kineticFolder,
                 resultsFolder,
                 elements_list=['C', 'H', 'O'],
                 threshold= 0.05,
                 ) 


