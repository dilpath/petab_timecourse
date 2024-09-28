import copy
from itertools import chain
from typing import Any, Dict, List, Sequence, Tuple, Union
import warnings

import amici
from amici.parameter_mapping import ParameterMapping
from amici.petab.conditions import create_edatas
from more_itertools import one, only
import numpy as np
import pandas as pd
import petab.v1
from petab.v1.C import (
    CONDITION_NAME,
    SIMULATION_CONDITION_ID,
    TIME,
    MEASUREMENT,
    OBSERVABLE_ID,
)
from pypesto.objective.amici.amici_util import create_identity_parameter_mapping

#from .misc import (
#    #get_timecourse,
#    subset_petab_problem,
#)



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
    amici_models: list[amici.Model],
    petab_problems: list[petab.v1.Problem],
) -> List[amici.ExpData]:
    """Precreate AMICI ExpData objects for PEtab problems with one AMICI model.

    As this is for a timecourse, which only simulates one condition at a time,
    this is a list of one AMICI ExpData object per timecourse period.

    Args:
        amici_models:
            The AMICI model, for each PEtab problem.
        petab_problems:
            The PEtab problems.

    Returns:
        The AMICI ExpData objects. The outer list is over PEtab problems,
        the inner list is over PEtab problem conditions.
    """
    edata_periods = []
    for amici_model, petab_problem in zip(amici_models, petab_problems, strict=True):
        # TODO precreate simulation conditions?
        edata = only(
            create_edatas(
                amici_model=amici_model,
                petab_problem=petab_problem,
            ),
            amici.ExpData(amici_model),
        )
        edata_periods.append(edata)
    return edata_periods


def precreate_parameter_mapping_periods(
    amici_models: amici.Model,
    petab_problem0: petab.v1.Problem,
    petab_problems: List[petab.v1.Problem],
    experiment_id: str = None,
) -> List[List[ParameterMapping]]:
    """Precreate AMICI parameter mapping objects for PEtab problems.

    NB: The parameter mapping will be for unscaled (linear) values.

    Args:
        amici_models:
            The period-specific AMICI model.
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
        petab.v1.get_optimization_to_simulation_parameter_mapping(
            condition_df=petab_problem0.condition_df,
            #experiment_df=petab_problem0.experiment_df,
            measurement_df=petab_problem0.measurement_df,
            parameter_df=petab_problem0.parameter_df,
            observable_df=petab_problem0.observable_df,
            mapping_df=petab_problem0.mapping_df,
            model=petab_problem0.model,
            simulation_conditions=pd.DataFrame(data={SIMULATION_CONDITION_ID: [experiment_id]}),
            scaled_parameters=True,
        )
    )

    for amici_model, petab_problem in zip(amici_models, petab_problems, strict=True):
        # Create dummy measurement df, for timecourse periods
        # that happen to have no measurements.
        # This is a quickfix to ensure that things like parameter
        # scales etc. are in the parameter mapping.
        # FIXME check if gradients etc are still correct with this
        #       should be OK since the dummy measurements aren't
        #       included in the likelihood
        dummy_petab_problem = copy.deepcopy(petab_problem)
        if dummy_petab_problem.measurement_df.empty:
            dummy_petab_problem.measurement_df = pd.concat(
                [
                    dummy_petab_problem.measurement_df,
                    pd.DataFrame(data={
                        OBSERVABLE_ID: [
                            dummy_petab_problem
                            .observable_df
                            .iloc[0]
                            .name
                        ],
                        SIMULATION_CONDITION_ID: [experiment_id],
                        TIME: [0.1],
                        MEASUREMENT: [0.1],
                    }),
                ],
                ignore_index=True,
            )
        parameter_mapping = amici.petab.parameter_mapping.ParameterMapping()
        parameter_mapping = amici.petab.parameter_mapping.create_parameter_mapping(
            petab_problem=dummy_petab_problem,
            simulation_conditions=[{SIMULATION_CONDITION_ID: one(dummy_petab_problem.measurement_df.loc[:, SIMULATION_CONDITION_ID].unique())}],
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
