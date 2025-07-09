# ✅ TASK 4: Product Mix Optimization using PuLP

# ✅ STEP 1: INSTALL & IMPORT LIBRARIES
!pip install pulp --quiet

from pulp import *
import pandas as pd

# ✅ STEP 2: PROBLEM DESCRIPTION
"""
A manufacturer produces 2 products: A and B.
- Each unit of Product A requires: 2 hours labor, 3 units material
- Each unit of Product B requires: 1 hour labor, 2 units material
- Profit: $40 per unit of A, $30 per unit of B
- Constraints:
  • Maximum 100 labor hours
  • Maximum 120 units of raw material

Goal: Determine how many units of A and B to produce to **maximize total profit**
"""

# ✅ STEP 3: DEFINE LP PROBLEM
prob = LpProblem("Product_Mix_Optimization", LpMaximize)

# Decision Variables
A = LpVariable("Product_A", lowBound=0, cat='Integer')
B = LpVariable("Product_B", lowBound=0, cat='Integer')

# ✅ STEP 4: OBJECTIVE FUNCTION
prob += 40 * A + 30 * B, "Total_Profit"

# ✅ STEP 5: CONSTRAINTS
prob += 2 * A + 1 * B <= 100, "Labor_Hours"
prob += 3 * A + 2 * B <= 120, "Raw_Material"

# ✅ STEP 6: SOLVE THE PROBLEM
prob.solve()

# ✅ STEP 7: PRINT RESULTS
print(f"Status: {LpStatus[prob.status]}")
print(f"Optimal units of Product A: {A.varValue}")
print(f"Optimal units of Product B: {B.varValue}")
print(f"Maximum Profit: ${value(prob.objective)}")
