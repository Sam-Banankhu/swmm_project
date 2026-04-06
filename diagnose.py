"""
diagnose.py
===========
Run this from your swmm_project folder to find out exactly why SWMM
is still rejecting the generated scenario file.

    python diagnose.py

It will:
  1. Print the first 60 lines of Example8.inp around [POLLUTANTS] and [INFLOWS]
  2. Build a test scenario .inp and print the relevant sections
  3. Try to open it in SWMM and show the full .rpt error output
  4. Print a clear diagnosis
"""

import os, sys, re

HERE = os.path.dirname(os.path.abspath(__file__))
INP  = os.path.join(HERE, 'Example8.inp')
PY   = os.path.join(HERE, 'dataset_generator.py')

for p in [INP, PY]:
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found. Run from your swmm_project folder.")

def read(p):
    with open(p, encoding='utf-8', errors='replace') as f:
        return f.read()

sep = "=" * 60

# ── 1. Inspect Example8.inp ───────────────────────────────────────────────────
print(f"\n{sep}")
print("STEP 1: Sections in Example8.inp")
print(sep)

inp = read(INP)
sections = re.findall(r'^\[.*?\]', inp, re.MULTILINE)
print("  Sections found:", sections)
print()

print("  [POLLUTANTS] present:", '[POLLUTANTS]' in inp)
if '[POLLUTANTS]' in inp:
    block = inp[inp.find('[POLLUTANTS]'):inp.find('\n[', inp.find('[POLLUTANTS]')+1)]
    print("  [POLLUTANTS] block:")
    for line in block.splitlines():
        print(f"    {repr(line)}")

print()
print("  Last 3 lines of [INFLOWS]:")
inflows = inp[inp.find('[INFLOWS]'):inp.find('\n[', inp.find('[INFLOWS]')+1)]
for line in inflows.splitlines()[-3:]:
    print(f"    {repr(line)}")

# ── 2. Build a scenario .inp ──────────────────────────────────────────────────
print(f"\n{sep}")
print("STEP 2: Building test scenario .inp")
print(sep)

CARRIER_FLOW = 0.01

def fmt_time(hours):
    hours = min(max(hours, 0.0), 11.99)
    h, m = int(hours), int(round((hours % 1) * 60))
    if m == 60: h += 1; m = 0
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

content = inp  # use the actual Example8.inp on disk

if '[POLLUTANTS]' not in content:
    print("  NOTE: [POLLUTANTS] missing from Example8.inp -- injecting now")
    pb = ('[POLLUTANTS]\n'
          ';;Name           Units  Crain  Cgw    Crdii  Kdecay SFflag\n'
          ';;-------------- ------ ------ ------ ------ ------ ------\n'
          'CONTAM           MG/L   0.0    0.0    0.0    0.0    NO\n\n')
    content = content.replace('[INFLOWS]', pb + '[INFLOWS]')
else:
    print("  [POLLUTANTS] already in Example8.inp -- using as-is")

# Check for the line endings issue
if '\r\n' in content:
    print("  NOTE: Example8.inp has Windows line endings (CRLF)")
else:
    print("  Line endings: Unix (LF)")

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

print(f"  ts_done={ts_done}, inflow_done={inf_done}")
if not ts_done:
    print("  WARNING: timeseries anchor line 'Rain_023in ... 12:00' NOT found!")
if not inf_done:
    print("  WARNING: inflow anchor line 'J12              FLOW ... 0.0125' NOT found!")

tmp_inp = os.path.join(HERE, '_diagnose_test.inp')
tmp_rpt = tmp_inp.replace('.inp', '.rpt')
tmp_out = tmp_inp.replace('.inp', '.out')
with open(tmp_inp, 'w') as f:
    f.writelines(out)
print(f"  Written: {tmp_inp}")

# Print key sections of the generated file
generated = ''.join(out)
print()
print("  [POLLUTANTS] block in generated file:")
if '[POLLUTANTS]' in generated:
    block = generated[generated.find('[POLLUTANTS]'):generated.find('\n[', generated.find('[POLLUTANTS]')+1)]
    for line in block.splitlines():
        print(f"    {repr(line)}")
else:
    print("    MISSING!")

print()
print("  CONTAM lines in [INFLOWS]:")
inflows_gen = generated[generated.find('[INFLOWS]'):generated.find('\n[', generated.find('[INFLOWS]')+1)]
for line in inflows_gen.splitlines():
    if 'CONTAM' in line or source_node in line:
        print(f"    {repr(line)}")

# ── 3. Try SWMM ───────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("STEP 3: Opening generated .inp with SWMM")
print(sep)

try:
    from swmm.toolkit.solver import swmm_open, swmm_close
except ImportError:
    sys.exit("  swmm-toolkit not installed. Run: pip install pyswmm swmm-toolkit")

try:
    swmm_open(tmp_inp, tmp_rpt, tmp_out)
    swmm_close()
    print("  swmm_open: SUCCESS")
except Exception as e:
    print(f"  swmm_open: FAILED -- {e}")

# Always print the rpt
if os.path.exists(tmp_rpt):
    rpt = open(tmp_rpt, errors='replace').read().strip()
    if rpt:
        print()
        print("  Full .rpt output:")
        for line in rpt.splitlines():
            print(f"    {line}")
    else:
        print("  .rpt file is empty (error occurred before SWMM could write anything)")
        print()
        print("  This usually means the .inp file has a structural problem that")
        print("  SWMM cannot even begin to parse. Check for:")
        print("    - Corrupted or truncated Example8.inp")
        print("    - Wrong file encoding (try opening Example8.inp in Notepad and re-saving as UTF-8)")
        print("    - A [POLLUTANTS] block with extra columns after SFflag=NO")
else:
    print("  .rpt file was not created at all.")

# ── Cleanup ───────────────────────────────────────────────────────────────────
for f in [tmp_inp, tmp_rpt, tmp_out]:
    try: os.remove(f)
    except: pass

print(f"\n{sep}")
print("DIAGNOSIS COMPLETE")
print(sep)
print()
print("Please paste the full output above into the chat.")