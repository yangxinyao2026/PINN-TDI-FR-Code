import pathlib
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
from Simulator.solver_environment import configure_ipopt

# Unified for all directly runSimulatorScript configurationConda/Ipopt DLLsearch path.
# Package import remains permissive; entry points that require Ipopt validate it explicitly.
IPOPT_EXECUTABLE = configure_ipopt()
#print(f'PROJECT_ROOT: {PROJECT_ROOT}')

