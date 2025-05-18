# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 01:49:46 2025

@author: imtia
"""

import random
from gurobipy import Model, GRB, quicksum

# Sets
S = ['S1', 'S2', 'S3', 'S4']  # Suppliers
M = ['MP1', 'MP2', 'MP3', 'MP4', 'MP5']  # Mobile pharmacies
B = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9', 'CL10']  # Candidate locations
H = ['H1', 'H2', 'H3']  # Hospitals
D = ['AF1', 'AF2', 'AF3']  # Affected areas
P = ['PI1', 'PI2', 'PI3', 'PI4']  # Pharmaceutical items
Omega = ['low', 'medium', 'high']  # Scenarios
pi = {'low': 0.2, 'medium': 0.5, 'high': 0.3}  # Scenario probabilities

# Distance data from candidate locations to affected areas (in km)
distances = {
    # AF1
    ('AF1', 'CL1'): 0.8,    ('AF1', 'CL2'): 0.024, ('AF1', 'CL3'): 2.3,
    ('AF1', 'CL4'): 6.2,    ('AF1', 'CL5'): 7.1,   ('AF1', 'CL6'): 11.4,
    ('AF1', 'CL7'): 11.9,   ('AF1', 'CL8'): 25.1,  ('AF1', 'CL9'): 2.4,
    ('AF1', 'CL10'): 29.9,
    
    # AF2
    ('AF2', 'CL1'): 5.7,    ('AF2', 'CL2'): 2.2,   ('AF2', 'CL3'): 1.2,
    ('AF2', 'CL4'): 4.1,    ('AF2', 'CL5'): 2.9,   ('AF2', 'CL6'): 2.1,
    ('AF2', 'CL7'): 9.4,    ('AF2', 'CL8'): 22.6,  ('AF2', 'CL9'): 0.067,
    ('AF2', 'CL10'): 31.2,
    
    # AF3
    ('AF3', 'CL1'): 25.4,   ('AF3', 'CL2'): 24.2,  ('AF3', 'CL3'): 23.3,
    ('AF3', 'CL4'): 27.2,   ('AF3', 'CL5'): 22.2,  ('AF3', 'CL6'): 26.9,
    ('AF3', 'CL7'): 14.2,   ('AF3', 'CL8'): 0.75,  ('AF3', 'CL9'): 24.2,
    ('AF3', 'CL10'): 19.8,
}

# Weights for multi-objective - less weight on strategic costs
w1 = 0.7  # operational cost
w2 = 0.3  # strategic cost

model = Model("MultiObjective_Stochastic_Pharmaceutical_Distribution")

# Parameters
f_s = {'S1': 1134, 'S2': 1847, 'S3': 1763, 'S4': 1255}
f_m_extra = 1600

# Mobile pharmacy deployment costs
mp_deploy_cost = {'MP1': 1800, 'MP2': 2000, 'MP3': 1800, 'MP4': 2000, 'MP5': 2000}

gamma_sp = {
    ('S1', 'PI1'): 6588, ('S1', 'PI2'): 6071, ('S1', 'PI3'): 6481, ('S1', 'PI4'): 5436,
    ('S2', 'PI1'): 5244, ('S2', 'PI2'): 5596, ('S2', 'PI3'): 6067, ('S2', 'PI4'): 5939,
    ('S3', 'PI1'): 6227, ('S3', 'PI2'): 6099, ('S3', 'PI3'): 6025, ('S3', 'PI4'): 6308,
    ('S4', 'PI1'): 6787, ('S4', 'PI2'): 6725, ('S4', 'PI3'): 6757, ('S4', 'PI4'): 5594
}

theta_sb = {(s, b): 70 for s in S for b in B}

zeta_bd = {}
for b in B:
    for d in D:
        if (d, b) in distances:
            zeta_bd[(b, d)] = distances[(d, b)]
        else:
            zeta_bd[(b, d)] = 50

pc_ic = {
    ('S1', 'PI2'): 247, ('S1', 'PI3'): 227,
    ('S2', 'PI2'): 372, ('S2', 'PI3'): 218,
    ('S3', 'PI2'): 346, ('S3', 'PI3'): 214,
    ('S4', 'PI2'): 232, ('S4', 'PI3'): 232
}
i_sp = {(s, p): pc_ic.get((s, p), 250) for s in S for p in P}

xi_mab = {}
for m in M:
    for a in B:
        for b in B:
            if a != b:
                direct_dist = None
                for d in D:
                    if (d, a) in distances and (d, b) in distances:
                        dist_a = distances[(d, a)]
                        dist_b = distances[(d, b)]
                        direct_dist = abs(dist_a - dist_b)
                        break
                xi_mab[(m, a, b)] = direct_dist if direct_dist is not None else 50
            else:
                xi_mab[(m, a, b)] = 100000

epsilon_m = {'MP1': 1800, 'MP2': 2000, 'MP3': 1800, 'MP4': 2000, 'MP5': 2000}
delivery_limit_m = {'MP1': 500, 'MP2': 500, 'MP3': 400, 'MP4': 200, 'MP5': 300}
scen_demands = {'low': {'hospital': 80, 'area': 120}, 
                'medium': {'hospital': 100, 'area': 150}, 
                'high': {'hospital': 120, 'area': 180}}
rho_ht = {h: 1000 for h in H}
rho_dt = {d: 1500 for d in D}
r = 0.8

# Decision Variables
x = model.addVars(S, vtype=GRB.BINARY, name="x")
x_m = model.addVars(M, vtype=GRB.BINARY, name="x_m")
alpha = model.addVars(M, B, B, Omega, vtype=GRB.BINARY, name="alpha")
y = model.addVars(M, B, Omega, vtype=GRB.BINARY, name="y")
delta_mbo = model.addVars(M, B, Omega, vtype=GRB.BINARY, name="delta_mbo")
eta_h = model.addVars(H, Omega, vtype=GRB.BINARY, name="eta_h")
eta_d = model.addVars(D, Omega, vtype=GRB.BINARY, name="eta_d")
mu = model.addVars(S, H, P, Omega, lb=0, name="mu")
lam_sb = model.addVars(S, B, P, Omega, lb=0, name="lambda_sb")
lam_bd = model.addVars(B, D, P, Omega, lb=0, name="lambda_bd")

# Objective Function - Multi-objective with operational and strategic costs
model.setObjective(
    # Operational costs (w1)
    w1 * (
        quicksum(pi[o] * (
            quicksum(i_sp[s, p] * mu[s, h, p, o] for s in S for h in H for p in P) +
            quicksum(theta_sb[s, b] * lam_sb[s, b, p, o] for s in S for b in B for p in P) +
            quicksum(zeta_bd[b, d] * lam_bd[b, d, p, o] for b in B for d in D for p in P) +
            quicksum(xi_mab[m, a, b] * alpha[m, a, b, o] for m in M for a in B for b in B)
        ) for o in Omega)
    ) +
    # Strategic costs (w2)
    w2 * (
        quicksum(f_s[s] * x[s] for s in S) +
        quicksum(mp_deploy_cost[m] * x_m[m] for m in M) +
        quicksum(pi[o] * (
            quicksum(f_m_extra * delta_mbo[m, b, o] for m in M for b in B)
        ) for o in Omega)
    ),
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
model.addConstr(quicksum(x_m[m] for m in M) == 2, name="Three_MPs_First_Stage")

for m in M:
    for o in Omega:
        for a in B:
            for b in B:
                if a != b:
                    model.addConstr(alpha[m, a, b, o] <= y[m, a, o], name=f"Movement_From_{m}_{a}_{b}_{o}")
                    model.addConstr(alpha[m, a, b, o] <= y[m, b, o], name=f"Movement_To_{m}_{a}_{b}_{o}")

# Solve
model.optimize()

# Output decisions
if model.status == GRB.OPTIMAL:
    print("\n========== OPTIMAL SOLUTION FOUND ==========")
    print("\nObjective Value: ${:,.2f}".format(model.objVal))
    print(f"Variables: {model.NumVars}, Constraints: {model.NumConstrs}")

    # Calculate and print cost breakdown
    operational_cost = 0
    strategic_cost = 0
    
    # Calculate operational costs
    for o in Omega:
        scenario_op_cost = (
            sum(i_sp[s, p] * mu[s, h, p, o].X for s in S for h in H for p in P) +
            sum(theta_sb[s, b] * lam_sb[s, b, p, o].X for s in S for b in B for p in P) +
            sum(zeta_bd[b, d] * lam_bd[b, d, p, o].X for b in B for d in D for p in P) +
            sum(xi_mab[m, a, b] * alpha[m, a, b, o].X for m in M for a in B for b in B)
        )
        operational_cost += pi[o] * scenario_op_cost
    
    # Calculate strategic costs
    supplier_selection_cost = sum(f_s[s] * x[s].X for s in S)
    mp_deployment_cost = sum(mp_deploy_cost[m] * x_m[m].X for m in M)
    extra_deploy_cost = sum(pi[o] * sum(f_m_extra * delta_mbo[m, b, o].X for m in M for b in B) for o in Omega)
    strategic_cost = supplier_selection_cost + mp_deployment_cost + extra_deploy_cost
    
    # Calculate weighted objective components
    weighted_operational = w1 * operational_cost
    weighted_strategic = w2 * strategic_cost
    total_cost = weighted_operational + weighted_strategic
    
    print("\nCost Breakdown:")
    print(f"Operational cost (weight {w1}): ${operational_cost:,.2f} (weighted: ${weighted_operational:,.2f})")
    print(f"Strategic cost (weight {w2}): ${strategic_cost:,.2f} (weighted: ${weighted_strategic:,.2f})")
    print(f"  - Supplier selection: ${supplier_selection_cost:,.2f}")
    print(f"  - Mobile pharmacy deployment: ${mp_deployment_cost:,.2f}")
    print(f"  - Extra deployment: ${extra_deploy_cost:,.2f}")
    print(f"Total weighted cost: ${total_cost:,.2f}")
    
    print("\n========== STAGE 1 DECISIONS ==========")
    print("\nSuppliers Selected:")
    for s in S:
        if x[s].X > 0.5:
            print(f"- {s}")
    
    print("\nMobile Pharmacies Deployed in Stage 1:")
    for m in M:
        if x_m[m].X > 0.5:
            print(f"- {m} (Deployment cost: ${mp_deploy_cost[m]:,.2f})")
    
    print("\n========== STAGE 2 DECISIONS BY SCENARIO ==========")
    for o in Omega:
        print(f"\n--- Scenario: {o.upper()} (Probability: {pi[o]*100}%) ---")
        
        # Calculate scenario-specific costs
        inventory_cost = sum(i_sp[s, p] * mu[s, h, p, o].X for s in S for h in H for p in P)
        transport_sl_cost = sum(theta_sb[s, b] * lam_sb[s, b, p, o].X for s in S for b in B for p in P)
        transport_la_cost = sum(zeta_bd[b, d] * lam_bd[b, d, p, o].X for b in B for d in D for p in P)
        movement_cost = sum(xi_mab[m, a, b] * alpha[m, a, b, o].X for m in M for a in B for b in B)
        scenario_extra_deploy = sum(f_m_extra * delta_mbo[m, b, o].X for m in M for b in B)
        
        scenario_op_cost = inventory_cost + transport_sl_cost + transport_la_cost + movement_cost
        scenario_total = scenario_op_cost + scenario_extra_deploy
        
        print(f"Scenario-specific costs:")
        print(f"  - Inventory cost: ${inventory_cost:,.2f}")
        print(f"  - Transportation cost (suppliers to locations): ${transport_sl_cost:,.2f}")
        print(f"  - Transportation cost (locations to areas): ${transport_la_cost:,.2f}")
        print(f"  - Mobile pharmacy movement cost: ${movement_cost:,.2f}")
        print(f"  - Extra deployment cost: ${scenario_extra_deploy:,.2f}")
        print(f"  - Total scenario cost: ${scenario_total:,.2f}")
        
        print("\nMobile Pharmacy Locations:")
        for m in M:
            for b in B:
                if y[m, b, o].X > 0.5:
                    deployed_type = "initially deployed" if x_m[m].X > 0.5 else "additionally deployed"
                    print(f"- {m} located at {b} ({deployed_type})")
        
        print("\nMobile Pharmacy Movements:")
        for m in M:
            for a in B:
                for b in B:
                    if a != b and alpha[m, a, b, o].X > 0.5:
                        dist = xi_mab[(m, a, b)]
                        print(f"- {m} moves from {a} to {b} (distance: {dist:.2f} km, cost: ${xi_mab[(m, a, b)]:,.2f})")
        
        print("\nSupplier to Hospital Flows:")
        for s in S:
            if x[s].X > 0.5:
                for h in H:
                    for p in P:
                        q = mu[s, h, p, o].X
                        if q > 1e-3:
                            print(f"- {q:.1f} units of {p} from {s} to {h}")
        
        print("\nSupplier to Mobile Pharmacy Flows:")
        for s in S:
            if x[s].X > 0.5:
                for b in B:
                    for p in P:
                        q = lam_sb[s, b, p, o].X
                        if q > 1e-3:
                            print(f"- {q:.1f} units of {p} from {s} to {b}")
        
        print("\nMobile Pharmacy to Affected Area Flows:")
        for b in B:
            for d in D:
                for p in P:
                    q = lam_bd[b, d, p, o].X
                    if q > 1e-3:
                        dist = distances.get((d, b), 'unknown')
                        print(f"- {q:.1f} units of {p} from {b} to {d} (distance: {dist} km, cost: ${zeta_bd[(b, d)]:,.2f})")
else:
    print("\nModel did not reach optimal solution. Status:", model.status)