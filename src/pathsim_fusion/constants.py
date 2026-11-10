"""
Numerical constants of the toolbox, each with the measurement it comes from.

The GLC measurements use GLC(P_in=2e5, L=1, D=0.1, T=623) with flow_l=1 and
flow_g=1e-4, where Bo_g ~ 3e5 and the gas transfer number is k ~ 2.3e-3/c_T_in.
"""

# Gas transfer number above which GLC uses the closed-form gas-limited limit
# instead of the BVP solve. The limit deviates from the solve by 3.2/k in the
# efficiency (C-C), 0.3% at k=1e3 where eff ~ 4e-5. The solve converges up to
# k ~ 2e4, so the switch keeps a margin of 20.
GLC_GAS_LIMITED_K = 1e3

# Lower bound of the gas inlet mole fraction scaled by the equilibrium ratio,
# w = y_T2 / nu. At a T2-free gas inlet under O-C the Sieverts square root is
# singular at xi=0 and the collocation does not reach its tolerance for
# c_T_in >= 1e-3. With this bound O-C converges for c_T_in from 1 to 2e-7 and
# the efficiency is unchanged to 9 digits.
GLC_W_IN_MIN = 1e-12

# Initial GLC mesh: geometric node spacing from GLC_MESH_EDGE_SPACING up to 0.1
# at both ends, for the gas inlet layer (width ~1/k) and the gas outlet layer
# (width ~1/Bo_g). Against an equidistant mesh this needs 3-8x fewer nodes and
# a tenth of the solve time, and the C-C solve converges up to k ~ 2e4 instead
# of 2e3.
GLC_MESH_EDGE_SPACING = 1e-8
GLC_MESH_EDGE_NODES = 30
