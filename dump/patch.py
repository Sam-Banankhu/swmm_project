"""
patch.py
========
Run this ONCE from your swmm_project folder to fix dataset_generator.py in-place.

    python patch.py

It patches build_scenario_inp() to inject a [POLLUTANTS] section into every
generated scenario .inp file.  Example8.inp is left completely unchanged.

Usage:
    cd C:\\Users\\samba\\Downloads\\swmm_example8_project (1)\\swmm_project
    python patch.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INP  = os.path.join(HERE, 'Example8.inp')
PY   = os.path.join(HERE, 'dataset_generator.py')

for p in [INP, PY]:
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found. Run this script from your swmm_project folder.")

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

def write(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  Written: {path}")


# ── Patch dataset_generator.py ────────────────────────────────────────────────

print("\n[1/2] Patching dataset_generator.py ...")

py = read(PY)
MARKER = "if '[POLLUTANTS]' not in content:"

if MARKER in py:
    print("  Already patched -- nothing to do.")
else:
    OLD = """    with open(base_inp) as f:
        lines = f.readlines()

    new_lines      = []
    ts_done        = False
    inflow_done    = False
    ts_name_flow   = f'CarrierFlow_{source_node}'
    ts_name_conc   = f'ContamConc_{source_node}'

    for line in lines:
        new_lines.append(line)

        # Append timeseries after the last Rain timeseries line
        if 'Rain_023in' in line and '12:00' in line and not ts_done:
            for t, v in ts_points(CARRIER_FLOW):
                new_lines.append(f'{ts_name_flow:<28}{t:<12}{v}\\n')
            for t, v in ts_points(concentration_mg_l):
                new_lines.append(f'{ts_name_conc:<28}{t:<12}{v}\\n')
            ts_done = True

        # Add inflow lines after the last dry weather inflow line
        if 'J12              FLOW' in line and '0.0125' in line and not inflow_done:
            new_lines.append(
                f'{source_node:<17}FLOW             {ts_name_flow:<17}DIRECT   1.0      1.0\\n'
            )
            new_lines.append(
                f'{source_node:<17}CONTAM           {ts_name_conc:<17}CONCEN   1.0      1.0\\n'
            )
            inflow_done = True

    with open(tmp_inp, 'w') as f:
        f.writelines(new_lines)"""

    NEW = """    ts_name_flow   = f'CarrierFlow_{source_node}'
    ts_name_conc   = f'ContamConc_{source_node}'

    with open(base_inp) as f:
        content = f.read()

    # FIX: Example8.inp is a hydraulics-only model with no [POLLUTANTS] section.
    # SWMM raises ERROR 209 ("undefined object CONTAM") if CONTAM appears in
    # [INFLOWS] without being declared as a pollutant first.
    # We inject a minimal [POLLUTANTS] block so every generated scenario .inp
    # is self-contained and valid.
    #
    # IMPORTANT: Do NOT add trailing optional columns (CoPoll CoFrac Cdwf Cinit)
    # after SFflag=NO.  SWMM 5.2 misreads bare zeros in those positions as object
    # references and raises another ERROR 209 ("undefined object 0.0").
    # The correct minimal declaration is:
    #   CONTAM    MG/L    0.0    0.0    0.0    0.0    NO
    if '[POLLUTANTS]' not in content:
        pollutants_block = (
            '[POLLUTANTS]\\n'
            ';;Name           Units  Crain  Cgw    Crdii  Kdecay SFflag\\n'
            ';;-------------- ------ ------ ------ ------ ------ ------\\n'
            'CONTAM           MG/L   0.0    0.0    0.0    0.0    NO\\n'
            '\\n'
        )
        content = content.replace('[INFLOWS]', pollutants_block + '[INFLOWS]')

    lines = content.splitlines(keepends=True)
    new_lines      = []
    ts_done        = False
    inflow_done    = False

    for line in lines:
        new_lines.append(line)

        # Append timeseries after the last Rain timeseries line
        if 'Rain_023in' in line and '12:00' in line and not ts_done:
            for t, v in ts_points(CARRIER_FLOW):
                new_lines.append(f'{ts_name_flow:<28}{t:<12}{v}\\n')
            for t, v in ts_points(concentration_mg_l):
                new_lines.append(f'{ts_name_conc:<28}{t:<12}{v}\\n')
            ts_done = True

        # Add inflow lines after the last dry weather inflow line
        if 'J12              FLOW' in line and '0.0125' in line and not inflow_done:
            new_lines.append(
                f'{source_node:<17}FLOW             {ts_name_flow:<17}DIRECT   1.0      1.0\\n'
            )
            new_lines.append(
                f'{source_node:<17}CONTAM           {ts_name_conc:<17}CONCEN   1.0      1.0\\n'
            )
            inflow_done = True

    with open(tmp_inp, 'w') as f:
        f.writelines(new_lines)"""

    if OLD not in py:
        print("  ERROR: Expected code block not found in dataset_generator.py.")
        print("  The file may already have been partially edited, or has Windows line endings.")
        print("  Restore the original from dataset_generator.py.bak if it exists,")
        print("  or re-download the original and try again.")
        sys.exit(1)

    bak = PY + '.bak'
    if not os.path.exists(bak):
        write(bak, py)
        print(f"  Backup saved: {bak}")

    py = py.replace(OLD, NEW)
    write(PY, py)
    print("  build_scenario_inp patched successfully.")
    print("  Example8.inp is unchanged (fix lives entirely in dataset_generator.py).")


# ── Smoke test ────────────────────────────────────────────────────────────────

print("\n[2/2] Running smoke test (scenario 1: source=J8, conc=994.9 mg/L) ...")

try:
    from swmm.toolkit.solver import swmm_open, swmm_start, swmm_step, swmm_end, swmm_close
    from swmm.toolkit import solver as slv
except ImportError:
    print("  swmm-toolkit not importable.  Install with:  pip install pyswmm swmm-toolkit")
    sys.exit(0)

CARRIER_FLOW = 0.01

def fmt_time(hours):
    hours = min(max(hours, 0.0), 11.99)
    h, m = int(hours), int(round((hours % 1) * 60))
    if m == 60:
        h += 1; m = 0
    return f"{h:02d}:{m:02d}"

source_node = 'J8'
conc = 994.9; dur = 0.32; start = 1.65
end_h = start + dur
pre_h = max(0.0, start - 1/60)
post_h = end_h + 5/60

def ts_pts(val):
    return [("00:00",0),(fmt_time(pre_h),0),(fmt_time(start),val),
            (fmt_time(end_h),val),(fmt_time(post_h),0),("12:00",0)]

tf = f'CarrierFlow_{source_node}'
tc = f'ContamConc_{source_node}'

with open(INP) as f:
    content = f.read()

if '[POLLUTANTS]' not in content:
    pb = ('[POLLUTANTS]\n'
          ';;Name           Units  Crain  Cgw    Crdii  Kdecay SFflag\n'
          ';;-------------- ------ ------ ------ ------ ------ ------\n'
          'CONTAM           MG/L   0.0    0.0    0.0    0.0    NO\n\n')
    content = content.replace('[INFLOWS]', pb + '[INFLOWS]')

lines = content.splitlines(keepends=True)
out = []; ts_done = inf_done = False
for line in lines:
    out.append(line)
    if 'Rain_023in' in line and '12:00' in line and not ts_done:
        for t, v in ts_pts(CARRIER_FLOW): out.append(f'{tf:<28}{t:<12}{v}\n')
        for t, v in ts_pts(conc):         out.append(f'{tc:<28}{t:<12}{v}\n')
        ts_done = True
    if 'J12              FLOW' in line and '0.0125' in line and not inf_done:
        out.append(f'{source_node:<17}FLOW             {tf:<17}DIRECT   1.0      1.0\n')
        out.append(f'{source_node:<17}CONTAM           {tc:<17}CONCEN   1.0      1.0\n')
        inf_done = True

tmp_inp = os.path.join(HERE, '_patch_smoke_test.inp')
tmp_rpt = tmp_inp.replace('.inp', '.rpt')
tmp_out = tmp_inp.replace('.inp', '.out')
with open(tmp_inp, 'w') as f:
    f.writelines(out)

nodes = []
section = None
with open(INP) as f:
    for line in f:
        l = line.strip()
        if not l or l.startswith(';;'): continue
        if l.startswith('['): section = l.strip('[]').upper(); continue
        parts = l.split()
        if parts and section in ('JUNCTIONS', 'OUTFALLS', 'STORAGE'):
            nodes.append(parts[0])

try:
    swmm_open(tmp_inp, tmp_rpt, tmp_out)
    swmm_start(True)
    peaks = {n: 0.0 for n in nodes}
    steps = 0
    while True:
        t = swmm_step()
        if t == 0: break
        steps += 1
        for i, n in enumerate(nodes):
            c = slv.node_get_pollutant(i, 0)[0]
            if c > peaks[n]: peaks[n] = c
    swmm_end()
    swmm_close()

    detected = {n: v for n, v in peaks.items() if v >= 5.0}
    print(f"  {steps} routing steps completed")
    print(f"  Nodes detecting >= 5 mg/L: {len(detected)}")
    for n, v in sorted(detected.items(), key=lambda x: -x[1]):
        print(f"    {n:<12}  {v:.2f} mg/L")
    print("\n  Patch verified. Now run:  python dataset_generator.py")

except Exception as e:
    rpt_text = open(tmp_rpt, errors='replace').read() if os.path.exists(tmp_rpt) else ''
    errors = [l.strip() for l in rpt_text.splitlines() if 'ERROR' in l]
    print(f"\n  SMOKE TEST FAILED: {e}")
    if errors:
        print(f"  RPT errors: {errors}")
    sys.exit(1)

finally:
    for f in [tmp_inp, tmp_rpt, tmp_out]:
        try: os.remove(f)
        except: pass