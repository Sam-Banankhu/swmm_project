SWMM Example 8 - Combined Sewer System
=======================================
Replicated for: Hybrid AI Sensor Placement Research
Authors: Mhango, S.B. and Sambito, M. (2026)

Source: EPA SWMM Applications Manual, Example 8 (pages 133-154)


HOW TO OPEN THIS FILE
---------------------
1. Install EPA SWMM 5.2 from:
   https://www.epa.gov/water-research/storm-water-management-model-swmm

2. Open SWMM

3. Go to File > Open

4. Select the file: Example8.inp

5. The network will appear on screen

6. Click the Run button (green lightning bolt) to run the simulation


WHAT IS IN THIS FILE
--------------------
This is a combined sewer system serving a 29-acre urban catchment.
It includes:

  - 7 subcatchments (S1 to S7)
  - 28 junction nodes (J1, J2, J3... and JI1, JI2, JI3...)
  - 2 outfalls (O1 = stream, O2 = WWTP)
  - 1 storage node (Well = pump wet well)
  - 10 stream conduits (C3 to C11, C_Aux3)
  - 6 combined sewer pipes (P1 to P6)
  - 9 interceptor pipes (I1 to I9)
  - 4 force main pipes (I10 to I13)
  - 5 flow regulators (R1 to R5, built from weirs W1-W4 and orifice Or1)
  - 1 pump (Pump1) with a Type 3 pump curve
  - Dry weather wastewater flows at J1, J2a, Aux3, J13, J12

The simulation runs for 12 hours using a 0.23-inch rainfall event.
This small storm produces no combined sewer overflows (CSOs), which
is useful for a first test run.


NODE NAMING CONVENTION
----------------------
  Jx   = combined sewer junctions (green pipes)
  JIx  = interceptor junctions (brown pipes)
  Aux3 = flow splitting node
  Well = pump wet well (storage node)
  O1   = stream outfall
  O2   = WWTP outfall


FOR THE ML RESEARCH
--------------------
The three nodes with elevated contamination probability are:
  J4, J10, and JI18

These nodes are considered to have double the baseline probability
of being a source of illicit discharge (Sambito et al., 2020).

When you start generating contamination scenarios for ML training,
these nodes should be sampled at twice the rate of other nodes.


ADDING WATER QUALITY (for ML data generation)
----------------------------------------------
To add a contamination event manually:

1. Go to Project > Pollutants > Add
   Name: CONTAM
   Units: MG/L
   Decay coefficient: 0 (conservative pollutant)

2. Click on a junction node (e.g. J4)
   Open its properties
   Go to Inflows > Direct Inflow tab
   Set Constituent = CONTAM
   Create a time series that spikes to a concentration value
   then returns to zero

3. Run the simulation and view the concentration graph
   at any node by right-clicking it > Graph > Quality


REFERENCE
---------
Sambito, M., Di Cristo, C., Freni, G., and Leopardi, A. (2020).
Optimal water quality sensor positioning in urban drainage systems
for illicit intrusion identification.
Journal of Hydroinformatics, 22(1), 46-60.
https://doi.org/10.2166/hydro.2019.036
