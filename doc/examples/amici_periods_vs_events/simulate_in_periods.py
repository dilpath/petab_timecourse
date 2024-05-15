# # Simple timecourse
# 
# In this example, a simple timecourse is defined for a simple ODE. Simulation in periods is demonstrated.
# 
# The ODE is $\frac{d}{dt} x = p  q$, with $p$ an estimated parameter, and $q$ a timecourse parameter.
# 
# NB: in code, symbols are suffixed with `_` to avoid conflicts with packages.

from itertools import chain
from pathlib import Path

import amici
import matplotlib.pyplot as plt
import numpy as np
import petab

import petab_timecourse
from petab_timecourse.amici import collect_x, collect_sx, collect_t
from petab_timecourse.simulator import AmiciSimulator

from _helpers import get_analytical_x_, get_analytical_sx_


petab_path = Path('input') / 'simple_timecourse'
output_path = Path('output')
plot_output_path = output_path / 'plot'
plot_output_path.mkdir(parents=True, exist_ok=True)

timecourse_id = 'timecourse1'
true_p_ = 1
petab_problem = petab.Problem.from_yaml(str(petab_path / 'petab.yaml'))

# Setup timecourse, add it to the PEtab problem
timecourse_df = petab_timecourse.get_timecourse_df(petab_path / 'timecourse.tsv')
petab_problem.timecourse_df = timecourse_df
timecourse = petab_timecourse.Timecourse.from_df(timecourse_df, timecourse_id, petab_problem.condition_df)

# Setup AMICI
solver_settings = {
    'setSensitivityOrder': amici.SensitivityOrder_first,
    'setSensitivityMethod': amici.SensitivityMethod_forward,
    'setMaxSteps': int(1e6),
    'setMaxTime': 60,
}

amici_model = amici.petab_import.import_petab_problem(petab_problem, model_output_dir="amici_model_events")
amici_solver = amici_model.getSolver()
for setter, value in solver_settings.items():
    getattr(amici_solver, setter)(value)

# Simulate in periods.
simulator = AmiciSimulator(
    petab_problem=petab_problem,
    timecourse_id=timecourse_id,
    amici_model=amici_model,
    amici_solver=amici_solver,
)
results = simulator.simulate()

# Collect results from each timecourse period
x_ = petab_timecourse.amici.collect_x(results)
sx_ = petab_timecourse.amici.collect_sx(results)
T = petab_timecourse.amici.collect_t(results)

# Plot against expected simulation.
analytical_x_  = [np.round(get_analytical_x_(t, timecourse=timecourse, p_=true_p_), 5)  for t in T]
analytical_sx_ = [np.round(get_analytical_sx_(t, timecourse=timecourse), 5) for t in T]

sx_ = np.squeeze(sx_)

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(T, x_, lw=5, label='Simulated state')
ax.plot(T, analytical_x_, linestyle=':', lw=10, label='Analytical state')
ax.legend()
ax.set_title('State trajectory');
plt.savefig(str(plot_output_path / 'x.png'))

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(T, sx_, lw=5, label='Simulated state sensitivity')
ax.plot(T, analytical_x_, linestyle=':', lw=10, label='Analytical state sensitivity')
ax.legend()
ax.set_title('State sensitivity trajectory');
plt.savefig(str(plot_output_path / 'sx.png'))
