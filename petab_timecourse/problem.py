# TODO rename times to timepoints?
# TODO initial assignments in SBML for states are not supported. Should be
#      automatically detected and remove, or issue a warning?

import copy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, Union

import amici
import numpy as np
import pandas as pd
import petab
from petab.C import (
    CONDITION_ID,
    CONDITION_NAME,
    TIME,
    TIME_STEADY_STATE,
)

from .petab import Condition
from .C import (
    NON_COMPONENT_CONDITION_LABELS,
    TYPE_CONDITION_VALUE,
    TYPE_TIME,
    TIMECOURSE_ID,
    TIMECOURSE,
    TIMECOURSE_NAME,
    TIME_CONDITION_DELIMITER,
    PERIOD_DELIMITER,
)
from .format import get_timecourse_df

from .sbml import add_timecourse_as_events


class Problem:
    @staticmethod
    def from_yaml(*args, **kwargs):
        problem = petab.Problem.from_yaml(*args, **kwargs)

        timecourse_files = problem.extensions_config['timecourse']['timecourse_files']
        if len(timecourse_files) > 1:
            raise ValueError("multiple timecourse files are not yet supported.")
        if len(timecourse_files) < 1:
            raise ValueError("no timecourse files?")
        timecourse_df = get_timecourse_df(timecourse_files[0])

        problem.timecourse_df = timecourse_df
        return problem


def get_models(problem: petab.Problem):
    models = {}
    for timecourse_id in problem.timecourse_df.index:
        models[timecourse_id] = add_timecourse_as_events(
            problem,
            timecourse_id=timecourse_id,
        )
    return models
