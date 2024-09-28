from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import petab
from petab.v1.C import (
    OBSERVABLE_ID,
    MEASUREMENT,
    SIMULATION_CONDITION_ID,
    TIME,
    OBSERVABLE_FORMULA,
    NOISE_FORMULA,
    NOMINAL_VALUE,
    ESTIMATE,
    PARAMETER_ID,
)
from petab.v2.C import (
    EXPERIMENT_ID,
    EXPERIMENT,
)

from .C import (
    DUMMY_OBSERVABLE_ID,
    DUMMY_MEASUREMENT,
    DUMMY_NOISE,
    TYPE_PATH,
    TYPE_TIME,
    TIME_CONDITION_DELIMITER,
    PERIOD_DELIMITER,
    TIMECOURSE,
    # FIXME: Usage of this in this file overlaps with the `Condition` class in
    # `.petab`
    NON_COMPONENT_CONDITION_LABELS,
)


#def get_path(path_like: TYPE_PATH) -> Path:
#    return Path(path_like)
#
#
#def times_to_durations(times: Iterable[float]) -> List[float]:
#    return [
#        times[i] - (times[i-1] if i != 0 else 0)
#        for i, _ in enumerate(times)
#    ]
#
#
#def parse_timecourse_string(
#    timecourse_string: str,
#) -> List[Tuple[TYPE_TIME, List[str]]]:
#    return [
#        time_condition.split(TIME_CONDITION_DELIMITER)
#        for time_condition in timecourse_string.split(PERIOD_DELIMITER)
#    ]
#
#
#def get_timecourse(petab_problem: petab.Problem, timecourse_id: str):
#    return parse_timecourse_string(
#        petab_problem.timecourse_df.loc[timecourse_id][TIMECOURSE],
#    )
#
#
#def parse_timecourse_string_as_lists(
#        timecourse_string: str,
#) -> Tuple[List[TYPE_TIME], List[str]]:
#    # FIXME unify with above function
#    result = parse_timecourse_string(timecourse_string)
#    timepoints = []
#    condition_ids = []
#    for entry in result:
#        timepoints.append(entry[0])
#        condition_ids.append(entry[1])
#    return timepoints, condition_ids


def subset_petab_problem(
    petab_problem: petab.Problem,
    experiment_id: str,
) -> list[petab.Problem]:
    """Split a PEtab problem into period-specific problems.

    Args:
        petab_problem:
            The PEtab problem.
        experiment_id:
            The ID of the experiment.

    Returns:
        The period-specific PEtab problems.
    """
    petab_problem0 = deepcopy(petab_problem)
    experiment = petab.Experiment.from_df(
        experiment_df=petab_problem.experiment_df,
        experiment_id=experiment_id,
    )
    petab_problems = []
    for period_index, period in enumerate(experiment.periods):
    #for index, (timepoint, condition_id) in enumerate(timecourse):
        petab_problem = deepcopy(petab_problem0)

        condition = petab_problem0.condition_df.loc[period.condition_id].to_dict()

        petab_problem.experiment_df = petab.get_experiment_df(pd.DataFrame(data={
            EXPERIMENT_ID: [],
            EXPERIMENT: [],
        }))

        # TODO create dummy PEtab v1 problem
        petab_problem.condition_df = petab.create_condition_df(parameter_ids=[])

        # Get period-specific measurements
        petab_problem.measurement_df = period.get_measurements(measurement_df=petab_problem0.measurement_df)
        petab_problem.measurement_df.loc[:, EXPERIMENT_ID] = None
        petab_problem.measurement_df.loc[:, SIMULATION_CONDITION_ID] = None

        # Fix condition parameters in parameter table.
        condition_parameter_df = petab.get_parameter_df(pd.DataFrame(data={
            PARAMETER_ID: condition.keys(),
            NOMINAL_VALUE: condition.values(),
            ESTIMATE: 1,
        }))
        petab_problem.parameter_df = pd.concat([
            petab_problem.parameter_df,
            condition_parameter_df,
        ])

        petab_problems.append(petab_problem)

    return petab_problems
