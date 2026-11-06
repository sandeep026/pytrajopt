import casadi as cs
from parse_toml import toml2sol

def meta():
    data = r"Ross, I. M., et al. “Scaling and Balancing for High-Performance Computation of Optimal Controls.” Journal of Guidance, Control, and Dynamics, vol. 41, no. 10, Oct. 2018, pp. 2086–97. DOI.org (Crossref), https://doi.org/10.2514/1.G003382."
    return data

def brach():
    prob_name='brachistochrone'
    toml2sol(prob_name,spar=True)

if __name__ == "__main__":
    brach()
