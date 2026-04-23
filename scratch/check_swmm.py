import os
from pyswmm import Simulation

inp = "./dataset/Examples/Example9_event.inp"
rpt = "test.rpt"
out = "test.out"

try:
    with Simulation(inp) as sim:
        for _ in sim:
            pass
    print("Base simulation worked!")
except Exception as e:
    print(f"Base simulation failed: {e}")
    if os.path.exists(rpt):
        with open(rpt) as f:
            print(f.read())
