import copy
from itertools import chain
from typing import Any, Dict, List, Sequence, Tuple, Union
import warnings

import amici
from amici.petab_import import import_petab_problem
from amici.parameter_mapping import ParameterMapping
from amici.petab_objective import (
    create_edatas,
    simulate_petab,
)
import amici.petab_objective
from more_itertools import one, only
import numpy as np
import petab
from petab.C import (
    CONDITION_NAME,
    SIMULATION_CONDITION_ID,
    TIME,
    MEASUREMENT,
    OBSERVABLE_ID,
)
from pypesto.objective.amici.amici_util import create_identity_parameter_mapping

from .misc import (
    #get_timecourse,
    subset_petab_problem,
)

def simulate_timecourse(
        parent_petab_problem: petab.Problem,
        timecourse_id: str,
        solver_settings: Dict[str, Any],
        problem_parameters: Dict[str, float] = None,
        parameter_mapping = None,
        sensi_orders: Tuple[int, ...] = (0,1),
        # FIXME Dict typehint
        initial_states: Union[Dict, Tuple[float, ...]] = None,
        model_settings: Dict[str, Any] = None,
        #solver_customizer: Callable[[amici.Solver], None] = None
):
    """
    Solver settings are a required attribute to ensure these are not
    forgotten...

    To use a default solver, supply an empty dictionary (`solver_settings={}`).

    initial_states:
        FIXME
        Can supply a dictionary or a tuple.
    """
    timecourse = get_timecourse(
        petab_problem=parent_petab_problem,
        timecourse_id=timecourse_id,
    )
    petab_problems = subset_petab_problem(
        petab_problem=parent_petab_problem,
        timecourse_id=timecourse_id,
    )
    #print([petab_problem.measurement_df for petab_problem in petab_problems])
    results = []
    #amici_model = import_petab_problem(petab_problems[0])
    for index, petab_problem in enumerate(petab_problems):
        if petab_problem.measurement_df.empty:
            # FIXME resolve this properly. Is it due to timecourse pieces
            #       that are defined to occur after the last measured time
            #       point?
            break
        # FIXME uncomment, remove global `amici_model = ...` above


        # Replace a controlled parameter with the ID of its timepoint-specific
        # control parameter (hack to ensure sensitivities are computed correctly)
        replace_timecourse_parameters = {
            k: v
            for k, v in petab_problem.condition_df.loc[timecourse_id].items()
            if isinstance(v, str) and k != CONDITION_NAME
        }
        petab_problem.condition_df.drop(
            replace_timecourse_parameters,
            axis=1,
            inplace=True,
        )
        for old_parameter_id, new_parameter_id in replace_timecourse_parameters.items():
            if old_parameter_id in problem_parameters:
                raise ValueError(
                    'A timecourse parameter was assigned a value via '
                    '`problem_parameters`, but should receive its value '
                    'from the timecourse information.'
                )
            if new_parameter_id not in problem_parameters:
                raise ValueError(
                    'Please supply a value for estimated timecourse '
                    'replacement parameters (parameter IDs in the rows) '
                    'of the condition table of a PEtab Timecourse problem.'
                )
            problem_parameters[old_parameter_id] = \
                problem_parameters[new_parameter_id]
            # Duplicate control parameter in parameter_df, new index value is the
            # controlled parameter, such that sensitivities are output for the
            # controlled parameter (and can be interpreted as the sensitivities for
            # the control parameter)
            petab_problem.parameter_df.loc[old_parameter_id] = \
                petab_problem.parameter_df.loc[new_parameter_id]

        amici_model = import_petab_problem(petab_problem)
        amici_model.setT0(float(timecourse[index][0]))

        amici_edatas = create_edatas(amici_model, petab_problem)
        if problem_parameters is not None:
            # FIXME temp fix to add into parameters that AMICI automatically
            #       sets to be estimated from the SBML model (that weren't
            #       fixed like parameters in a condition table)
            model_parameters = dict(zip(
                amici_model.getParameterIds(),
                amici_model.getParameters()
            ))

            if parameter_mapping is None:
                parameter_mapping = \
                    create_identity_parameter_mapping(amici_model, 1)
            # Remove parameters from problem parameters if they are already
            # specified by the timecourse.
            # May break if a problem parameter is named `'conditionName'`.
            subset_problem_parameters = {
                parameter_id: parameter_value
                for parameter_id, parameter_value in {
                    **model_parameters,
                    **problem_parameters
                }.items()
                if parameter_id not in petab_problem.condition_df.columns
            }
            removed_problem_parameters = \
                set(problem_parameters).difference(subset_problem_parameters)
            warnings.warn(
                'The following parameters were removed from the supplied '
                '`problem_parameters`, as they are already specified by the '
                f'timecourse: {removed_problem_parameters}'
            )

            amici.parameter_mapping.fill_in_parameters(
                edatas=amici_edatas,
                problem_parameters=subset_problem_parameters,
                scaled_parameters=True,
                parameter_mapping=parameter_mapping,
                amici_model=amici_model,
            )
        else:
            subset_problem_parameters = None

        if results:
            one(amici_edatas).tstart_ = timecourse.timepoints[index]
            one(amici_edatas).x0  = one(results[-1]['rdatas']).x[-1].flatten()
            one(amici_edatas).sx0 = one(results[-1]['rdatas']).sx[-1].flatten()
        elif initial_states is not None:
            # TODO untested
            if isinstance(initial_states, dict):
                indexed_initial_states = [
                    initial_states[state_id]
                    for state_id in amici_model.getStateIds()
                ]
            else:
                indexed_initial_states = initial_states
            one(amici_edatas).x0 = indexed_initial_states

        amici_solver = amici_model.getSolver()
        # TODO allow a user to specify these settings
        if model_settings is not None:
            for setter, value in model_settings.items():
                getattr(amici_model, setter)(value)
        if solver_settings is not None:
            for setter, value in solver_settings.items():
                getattr(amici_solver, setter)(value)

        results.append(simulate_petab(
            petab_problem=petab_problem,
            amici_model=amici_model,
            solver=amici_solver,
            edatas=amici_edatas,
            problem_parameters=subset_problem_parameters,
        ))

        if initial_states is not None:
            pass
        for old_parameter_id, new_parameter_id in replace_timecourse_parameters.items():
            results[-1]['sllh'][new_parameter_id] = \
                results[-1]['sllh'][old_parameter_id]
            # Was artifically added, so remove now
            del problem_parameters[old_parameter_id]
    return results


def collect_x(results):
    return np.concatenate(
        [
            rdata.x
            for result in results
            for rdata in result['rdatas']
        ],
        axis=0,
    )


def collect_sx(results):
    return np.concatenate(
        [
            rdata.sx
            for result in results
            for rdata in result['rdatas']
        ],
        axis=0,
    )


def collect_y(results):
    return np.concatenate(
        [
            rdata.y
            for result in results
            for rdata in result['rdatas']
        ],
        axis=0,
    )


def collect_sy(results):
    return np.concatenate(
        [
            rdata.sy
            for result in results
            for rdata in result['rdatas']
        ],
        axis=0,
    )


def collect_t(results):
    return np.concatenate(
        [
            rdata.ts
            for result in results
            for rdata in result['rdatas']
        ],
        axis=0,
    )


def remove_duplicates(T, *args):
    """Remove duplicated time points from AMICI results.

    Results are expected to be in the form provided by the methods
    - `collect_t`
    - `collect_x`
    - `collect_sx`

    Args:
        T:
            The vector of time with duplicates. Values at the indices
            corresponding to duplicates values in `T` will be removed from
            `T` and all other vectors. `T` is assumed to be sorted/monotonic.
        args:
            Other vectors that duplicates will be removed from.

    Returns:
        Deduplicated vectors, in the order provided in `args`.
    """
    t0 = None
    duplicated_indices = []
    for index, t in enumerate(T):
        if t0 is None:
            t0 = t
            continue
        if t == t0:
            duplicated_indices.append(index)
        t0 = t
    T = np.delete(T, obj=duplicated_indices, axis=0)
    deduplicated_vectors = [
        np.delete(vector, obj=duplicated_indices, axis=0)
        for vector in args
    ]
    return [T, *deduplicated_vectors]


def precreate_edata_periods(
    amici_model: amici.Model,
    petab_problems: List[petab.Problem],
) -> List[amici.ExpData]:
    """Precreate AMICI ExpData objects for PEtab problems with one AMICI model.

    As this is for a timecourse, which only simulates one condition at a time,
    this is a list of one AMICI ExpData object per timecourse period.

    Args:
        amici_model:
            The AMICI model.
        petab_problems:
            The PEtab problems.

    Returns:
        The AMICI ExpData objects. The outer list is over PEtab problems,
        the inner list is over PEtab problem conditions.
    """
    edata_periods = []
    for petab_problem in petab_problems:
        # TODO precreate simulation conditions?
        edata = only(
            amici.petab_objective.create_edatas(
                amici_model=amici_model,
                petab_problem=petab_problem,
            ),
            amici.ExpData(amici_model),
        )
        edata_periods.append(edata)
    return edata_periods


def precreate_parameter_mapping_periods(
    amici_model: amici.Model,
    petab_problem0: petab.Problem,
    petab_problems: List[petab.Problem],
    experiment_id: str = None,
) -> List[List[ParameterMapping]]:
    """Precreate AMICI parameter mapping objects for PEtab problems.

    NB: The parameter mapping will be for unscaled (linear) values.

    Args:
        amici_model:
            The governing model.
        petab_problems:
            The PEtab problems.
        experiment_id:
            The ID of the experiment.

    Returns:
        The AMICI parameter mapping objects. The outer list is over PEtab
        problems, the inner list is over PEtab problem conditions.
    """
    parameter_mapping_periods = []



    prelim_parameter_mapping = (
        petab.get_optimization_to_simulation_parameter_mapping(
            condition_df=petab_problem0.condition_df,
            experiment_df=petab_problem0.experiment_df,
            measurement_df=petab_problem0.measurement_df,
            parameter_df=petab_problem0.parameter_df,
            observable_df=petab_problem0.observable_df,
            mapping_df=petab_problem0.mapping_df,
            model=petab_problem0.model,
            simulation_experiments=[experiment_id],
            scaled_parameters=True,
        )
    )









    import ipdb; ipdb.set_trace()





    for petab_problem in petab_problems:
        # Create dummy measurement df, for timecourse periods
        # that happen to have no measurements.
        # This is a quickfix to ensure that things like parameter
        # scales etc. are in the parameter mapping.
        # FIXME check if gradients etc are still correct with this
        #       should be OK since the dummy measurements aren't
        #       included in the likelihood
        dummy_petab_problem = copy.deepcopy(petab_problem)
        if dummy_petab_problem.measurement_df.empty:
            dummy_petab_problem.measurement_df = (
                dummy_petab_problem.measurement_df.append(
                    {
                        OBSERVABLE_ID: (
                            dummy_petab_problem
                            .observable_df
                            .iloc[0]
                            .name
                        ),
                        SIMULATION_CONDITION_ID: timecourse_id,
                        TIME: 0.1,
                        MEASUREMENT: 0.1,
                    },
                    ignore_index=True,
                )
            )
        parameter_mapping = amici.petab.parameter_mapping.ParameterMapping()
        parameter_mapping = amici.petab_objective.create_parameter_mapping(
            petab_problem=dummy_petab_problem,
            simulation_conditions=[{SIMULATION_CONDITION_ID: one(petab_problem.measurement_df.loc[:, SIMULATION_CONDITION_ID].unique())}],
            scaled_parameters=True,
            amici_model=amici_model,
        )
        parameter_mapping_periods.append(parameter_mapping)
    return parameter_mapping_periods


def add_output_timepoints_if_missing(
    amici_edata: amici.ExpData,
    timepoints: List[float],
):
    all_timepoints = np.array(amici_edata.getTimepoints())
    all_data = list(amici_edata.getObservedData())
    all_data_std = list(amici_edata.getObservedDataStdDev())

    n_observables = amici_edata.nytrue()
    for timepoint in timepoints:
        if timepoint in all_timepoints:
            continue
        # AMICI timepoints must be sorted in ascending order,
        # find the position where the timepoint is greater than
        # the previous value, and lesser than the next value.
        if (timepoint < all_timepoints).all():
            timepoint_index = 0
        elif (timepoint > all_timepoints).all():
            timepoint_index = all_timepoints.size
        # Timepoint should be somewhere in the middle of the list
        else:
            timepoint_index = 1 + one(one(np.where(
                (all_timepoints < timepoint)[:-1] !=
                (all_timepoints < timepoint)[1:]
            )))
        # Insert timepoint and dummy data
        all_timepoints = np.insert(
            all_timepoints,
            timepoint_index,
            timepoint,
        )
        for _ in range(n_observables):
            all_data.insert(
                timepoint_index * n_observables,
                np.nan,
            )
            all_data_std.insert(
                timepoint_index * n_observables,
                np.nan,
            )

    amici_edata.setTimepoints(all_timepoints)
    amici_edata.setObservedData(all_data)
    amici_edata.setObservedDataStdDev(all_data_std)
