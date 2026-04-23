from pyswmm import Simulation, Links

inp = "./dataset/Examples/Example8.inp"
with Simulation(inp) as sim:
    links = Links(sim)
    for link in links:
        print(f"Link {link.linkid} type: {type(link)}")
        print(f"Attributes: {dir(link)}")
        try:
            print(f"Velocity: {link.velocity}")
        except Exception as e:
            print(f"Velocity failed: {e}")
        break
