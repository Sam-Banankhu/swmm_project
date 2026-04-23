import os
import shutil
from pyswmm import Simulation

# Copy Record.dat if not here
if not os.path.exists("Record.dat"):
    shutil.copy("dataset/Examples/Record.dat", "Record.dat")

# Manually create a failed scenario inp
base = "dataset/Examples/Example9_event.inp"
tmp = "failed_test.inp"
rpt = "failed_test.rpt"
out = "failed_test.out"

# Minimal injection logic test
with open(base) as f:
    content = f.read()

# 1. Fix Raingages
base_dir = os.path.dirname(os.path.abspath(base))
lines = content.split('\n')
new_lines = []
for line in lines:
    if line.strip() and not line.startswith('[') and not line.startswith(';'):
        parts = line.split()
        if 'FILE' in [p.upper() for p in parts]:
            f_idx = [p.upper() for p in parts].index('FILE')
            fname = parts[f_idx+1].strip('"')
            abs_fname = os.path.join(base_dir, fname)
            line = line.replace(fname, f'"{abs_fname}"')
    new_lines.append(line)
content = '\n'.join(new_lines)

# 2. Add Pollutant
content += "\n[POLLUTANTS]\nCONTAM MG/L 0 0 0 0 NO * 0 0 0\n"

# 3. Add Timeseries
content += "\n[TIMESERIES]\nTS1  00:00  1.0\nTS1  01:00  1.0\nTS1  02:00  0.0\n"

# 4. Add Inflow
content += "\n[INFLOWS]\nJ4 FLOW TS1 DIRECT 1.0 1.0\nJ4 CONTAM TS1 CONCEN 1.0 1.0\n"

with open(tmp, 'w') as f:
    f.write(content)

print("Running failed_test.inp ...")
try:
    with Simulation(tmp) as sim:
        for _ in sim:
            pass
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
    if os.path.exists(rpt):
        with open(rpt) as f:
            print("\n--- REPORT FILE ---")
            print(f.read())
