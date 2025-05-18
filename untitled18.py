# Reconstructing the full Gurobi multi-objective model code including solution printing

import random
from gurobipy import Model, GRB, quicksum

# Sets
S = ['S1', 'S2', 'S3', 'S4']
M = ['MP1', 'MP2', 'MP3', 'MP4', 'MP5']
B = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9', 'CL10']
H = ['H1', 'H2', 'H3']
D = ['AF1', 'AF2', 'AF3']
P = ['PI1', 'PI2', 'PI3', 'PI4']
Omega = ['low', 'medium', 'high']
pi = {'low': 0.2, 'medium': 0.5, 'high': 0.3}

# Weights for multi-objective
w1 = 0.4  # operational cost
w2 = 0.6  # penalty cost

model = Model("Stochastic_Pharmaceutical_Distribution")

# Parameters
#fixed agreement cost of the suppliers 
f_s = {'S1': 1134, 'S2': 1847, 'S3': 1763, 'S4': 1255}
#Mobile pharmacy deployment cost
f_m = {m: 1500 for m in M}
#Extar Establishment cost for mobile pharmacy m∈M (stage2
f_m_extra = 600

#Supply capacity of product p from supplier s
gamma_sp = {
    ('S1', 'PI1'): 6588, ('S1', 'PI2'): 6071, ('S1', 'PI3'): 6481, ('S1', 'PI4'): 5436,
    ('S2', 'PI1'): 5244, ('S2', 'PI2'): 5596, ('S2', 'PI3'): 6067, ('S2', 'PI4'): 5939,
    ('S3', 'PI1'): 6227, ('S3', 'PI2'): 6099, ('S3', 'PI3'): 6025, ('S3', 'PI4'): 6308,
    ('S4', 'PI1'): 6787, ('S4', 'PI2'): 6725, ('S4', 'PI3'): 6757, ('S4', 'PI4'): 5594
}
#Transportaion cost from supplier s to location b
#we have assumned a round cost
'''csm_io = {
    ('S1', 'CL1'): 85,
    ('S1', 'CL2'): 76,
    ('S1', 'CL3'): 77,
    ('S1', 'CL4'): 89,
    ('S1', 'CL5'): 90,
    ('S1', 'CL6'): 68,
    ('S1', 'CL7'): 62,
    ('S1', 'CL8'): 71,
    ('S1', 'CL9'): 68,
    ('S1', 'CL10'): 84,
    ('S2', 'CL1'): 67,
    ('S2', 'CL2'): 64,
    ('S2', 'CL3'): 67,
    ('S2', 'CL4'): 86,
    ('S2', 'CL5'): 71,
    ('S2', 'CL6'): 63,
    ('S2', 'CL7'): 87,
    ('S2', 'CL8'): 79,
    ('S2', 'CL9'): 78,
    ('S2', 'CL10'): 65,
    ('S3', 'CL1'): 83,
    ('S3', 'CL2'): 66,
    ('S3', 'CL3'): 73,
    ('S3', 'CL4'): 83,
    ('S3', 'CL5'): 79,
    ('S3', 'CL6'): 82,
    ('S3', 'CL7'): 61,
    ('S3', 'CL8'): 63,
    ('S3', 'CL9'): 88,
    ('S3', 'CL10'): 66,
    ('S4', 'CL1'): 68,
    ('S4', 'CL2'): 73,
    ('S4', 'CL3'): 60,
    ('S4', 'CL4'): 66,
    ('S4', 'CL5'): 80,
    ('S4', 'CL6'): 72,
    ('S4', 'CL7'): 70,
    ('S4', 'CL8'): 80,
    ('S4', 'CL9'): 60,
    ('S4', 'CL10'): 61,
}'''

theta_sb = {(s, b): 70 for s in S for b in B}

##Unit transportation cost between mobile pharmacy located at o and affected area l

zeta_bd = {(b, d): 10 for b in B for d in D}
'''cma_ol = {
    ('CL1', 'AF1'): 0.64,
    ('CL2', 'AF1'): 0.0192,
    ('CL3', 'AF1'): 1.84,
    ('CL4', 'AF1'): 4.96,
    ('CL5', 'AF1'): 5.68,
    ('CL6', 'AF1'): 9.12,
    ('CL7', 'AF1'): 9.52,
    ('CL8', 'AF1'): 20.08,
    ('CL9', 'AF1'): 1.92,
    ('CL10', 'AF1'): 23.92,
    ('CL1', 'AF2'): 4.56,
    ('CL2', 'AF2'): 1.76,
    ('CL3', 'AF2'): 0.96,
    ('CL4', 'AF2'): 3.28,
    ('CL5', 'AF2'): 2.32,
    ('CL6', 'AF2'): 1.68,
    ('CL7', 'AF2'): 7.52,
    ('CL8', 'AF2'): 18.08,
    ('CL9', 'AF2'): 0.0536,
    ('CL10', 'AF2'): 24.96,
    ('CL1', 'AF3'): 20.32,
    ('CL2', 'AF3'): 19.36,
    ('CL3', 'AF3'): 18.64,
    ('CL4', 'AF3'): 21.76,
    ('CL5', 'AF3'): 17.76,
    ('CL6', 'AF3'): 21.52,
    ('CL7', 'AF3'): 11.36,
    ('CL8', 'AF3'): 0.6,
    ('CL9', 'AF3'): 19.36,
    ('CL10', 'AF3'): 15.84,
}
'''
#Unit procurement cost of item p from supplier s items pi2 $ pi3 
pc_ic = {
    ('S1', 'PI2'): 247, ('S1', 'PI3'): 227,
    ('S2', 'PI2'): 372, ('S2', 'PI3'): 218,
    ('S3', 'PI2'): 346, ('S3', 'PI3'): 214,
    ('S4', 'PI2'): 232, ('S4', 'PI3'): 232
}

##Unit procurement cost of item p from supplier s items pi1 $ pi4 

i_sp = {(s, p): pc_ic.get((s, p), 250) for s in S for p in P}


#below dictionary represents the cost of moving a Mobile Pharmacy (MP) from one location to another.


xi_mab = {(m, a, b): 1 if a != b else 100000 for m in M for a in B for b in B}

#MP1 moves from CL4 to CL4 (same) then it is 10000
#This 10000 ensures logical MP movement across the network and prevents "wasting" a movement step to the same place.

##The storage capacity of each mobile pharmacy (MP) in terms of the maximum number of pharmaceutical items it can hold at any time

epsilon_m = {'MP1': 1800, 'MP2': 2000, 'MP3': 1800, 'MP4': 2000, 'MP5': 2000}
delivery_limit_m = {'MP1': 500, 'MP2': 500, 'MP3': 400, 'MP4': 200, 'MP5': 300}

scen_demands = {
    'low':    {'hospital': 80,  'area': 120},
    'medium': {'hospital': 100, 'area': 150},
    'high':   {'hospital': 120, 'area': 180}
}

rho_ht = {h: 1000 for h in H}
rho_dt = {d: 1500 for d in D}
#penalty cost
r = 0.8

# Decision Variables
x = model.addVars(S, vtype=GRB.BINARY, name="x")
#x Indicates whether supplier s is activated (1) or not (0)stage 1

x_m = model.addVars(M, vtype=GRB.BINARY, name="x_m")
# x_m Indicates whether Mobile Pharmacy m is deployed in Stage 1.

alpha = model.addVars(M, B, B, Omega, vtype=GRB.BINARY, name="alpha")
#alpha Equals 1 if MP m moves from location a to location b in scenario 

y = model.addVars(M, B, Omega, vtype=GRB.BINARY, name="y")
#y[m, b, o] – Binary Equals 1 if MP m is placed at base location b in scenario

delta_mbo = model.addVars(M, B, Omega, vtype=GRB.BINARY, name="delta_mbo")

#if MP m is deployed at b in scenario o and was not deployed in Stage 1.
#Comment: Helps compute extra setup cost for MPs added after uncertainty is revealed.

eta_h = model.addVars(H, Omega, vtype=GRB.BINARY, name="eta_h")


eta_d = model.addVars(D, Omega, vtype=GRB.BINARY, name="eta_d")
mu = model.addVars(S, H, P, Omega, lb=0, name="mu")
lam_sb = model.addVars(S, B, P, Omega, lb=0, name="lambda_sb")
lam_bd = model.addVars(B, D, P, Omega, lb=0, name="lambda_bd")

# Objective Function
model.setObjective(
    w1 * (
        quicksum(f_s[s] * x[s] for s in S) +
        quicksum(f_m[m] * x_m[m] for m in M) +
        quicksum(pi[o] * (
            quicksum(i_sp[s, p] * mu[s, h, p, o] for s in S for h in H for p in P) +
            quicksum(theta_sb[s, b] * lam_sb[s, b, p, o] for s in S for b in B for p in P) +
            quicksum(zeta_bd[b, d] * lam_bd[b, d, p, o] for b in B for d in D for p in P) +
            quicksum(xi_mab[m, a, b] * alpha[m, a, b, o] for m in M for a in B for b in B) +
            quicksum(f_m_extra * delta_mbo[m, b, o] for m in M for b in B)
        ) for o in Omega)
    ) +
    w2 * quicksum(pi[o] * (
        quicksum(rho_ht[h] * (1 - eta_h[h, o]) for h in H) +
        quicksum(rho_dt[d] * (1 - eta_d[d, o]) for d in D)
    ) for o in Omega),
    GRB.MINIMIZE
)

# Constraints
model.addConstrs((quicksum(mu[s, h, p, o] for h in H) + quicksum(lam_sb[s, b, p, o] for b in B) <= gamma_sp[s, p] * x[s]
    for s in S for p in P for o in Omega), name="Supplier_Capacity")
model.addConstrs((quicksum(mu[s, h, p, o] for s in S) >= scen_demands[o]['hospital'] * eta_h[h, o]
    for h in H for p in P for o in Omega), name="Hospital_Fulfillment")
model.addConstrs((quicksum(lam_bd[b, d, p, o] for b in B) >= r * scen_demands[o]['area'] * eta_d[d, o]
    for d in D for p in P for o in Omega), name="Affected_Coverage")
model.addConstrs((quicksum(y[m, b, o] for b in B) <= 1 for m in M for o in Omega), name="Mobile_Location_One")
model.addConstrs((delta_mbo[m, b, o] >= y[m, b, o] - x_m[m] for m in M for b in B for o in Omega), name="delta_indicator")
model.addConstrs((y[m, b, o] <= x_m[m] + delta_mbo[m, b, o] for m in M for b in B for o in Omega), name="MP_Location_If_Movement_Allowed")
model.addConstrs((quicksum(lam_bd[b, d, p, o] for d in D) <= quicksum(lam_sb[s, b, p, o] for s in S)
    for b in B for p in P for o in Omega), name="Flow_Conservation")
model.addConstrs((quicksum(lam_bd[b, d, p, o] for d in D for p in P) <= quicksum(epsilon_m[m] * y[m, b, o] for m in M)
    for b in B for o in Omega), name="Mobile_Capacity")
model.addConstrs((quicksum(lam_bd[b, d, p, o] * y[m, b, o] for b in B for d in D for p in P) <= delivery_limit_m[m]
    for m in M for o in Omega), name="MP_Delivery_Limit")
model.addConstrs((eta_h[h, o] == 1 for h in H for o in Omega), name="Force_Hospitals")
model.addConstrs((eta_d[d, o] == 1 for d in D for o in Omega), name="Force_AffectedAreas")
model.addConstr(quicksum(x_m[m] for m in M) == 3, name="Three_MPs_First_Stage")

# Solve
model.optimize()

# Output decisions
if model.status == GRB.OPTIMAL:
    print("\\n========== STAGE 1 DECISIONS ==========")
    for s in S:
        if x[s].X > 0.5:
            print(f"Supplier {s}: Selected")
    for m in M:
        if x_m[m].X > 0.5:
            deployed = False
            for o in Omega:
                for b in B:
                    if y[m, b, o].X > 0.5:
                        print(f"Mobile Pharmacy {m}: Deployed in Stage 1 at location {b}")
                        deployed = True
                        break
                if deployed:
                    break
            if not deployed:
                print(f"Mobile Pharmacy {m}: Deployed in Stage 1 but no location assigned")

    print("\\n========== STAGE 2 DECISIONS ==========")
    for o in Omega:
        print(f"\\n--- Scenario: {o.upper()} ---")
        for m in M:
            for b in B:
                if y[m, b, o].X > 0.5:
                    print(f"Mobile Pharmacy {m} located at {b} in scenario {o}")
        for m in M:
            for a in B:
                for b in B:
                    if alpha[m, a, b, o].X > 0.5:
                        print(f"{m} moves from {a} to {b} in scenario {o}")
        for s in S:
            for h in H:
                for p in P:
                    q = mu[s, h, p, o].X
                    if q > 1e-3:
                        print(f"{q:.1f} units of {p} from {s} to {h}")
        for s in S:
            for b in B:
                for p in P:
                    q = lam_sb[s, b, p, o].X
                    if q > 1e-3:
                        print(f"{q:.1f} units of {p} from {s} to {b}")
        for b in B:
            for d in D:
                for p in P:
                    q = lam_bd[b, d, p, o].X
                    if q > 1e-3:
                        print(f"{q:.1f} units of {p} from {b} to {d}")
else:
    print("\\nModel did not reach optimal solution.")

