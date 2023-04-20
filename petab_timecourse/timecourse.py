# TODO rename times to timepoints?
# TODO initial assignments in SBML for states are not supported. Should be
#      automatically detected and remove, or issue a warning?

import copy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union

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
from .misc import parse_timecourse_string_as_lists


class Period:
    """A timecourse period.

    Attributes:
        start:
            The start time.
        end:
            The end time.
        duration:
            The duration of the period.
        condition_id:
            The PEtab condition ID.
        original_petab_problem:
            The original PEtab problem that describes the full timecourse.
        petab_problem:
            The PEtab problem for this time period.
            e.g.: only contains measurements within the period start and end.
    """
    def __init__(self, duration: TYPE_TIME, condition_id: str):
        # Parameterized time (a parameter id) cannot be cast to float
        try:
            self.duration = float(duration)
        except ValueError:
            self.duration = duration

        self.condition_id = condition_id

    def get_condition(self, petab_problem: petab.Problem) -> pd.Series:
        return petab_problem.condition_df.loc[self.condition_id]

    def get_measurements(
        self,
        petab_problem: petab.Problem,
        t0: float,
        include_end: bool = False,
    ) -> pd.Series:
        after_start = petab_problem.measurement_df[TIME] >= t0
        before_end = (
            petab_problem.measurement_df[TIME] <= t0
            if include_end else
            petab_problem.measurement_df[TIME] < t0
        )
        return petab_problem.measurement_df.loc[after_start & before_end]

class Timecourse:
    """A timecourse.

    Attributes:
        timecourse_id:
            The timecourse ID.
        name:
            The timecourse name.
        periods:
            The periods of the timecourse.
    """
    def __init__(
        self,
        timecourse_id: str,
        periods: List[Period],
        name: Optional[str] = None,
        t0: float = None,
    ):
        self.timecourse_id = timecourse_id
        self.name = name
        self.periods = periods
        self.t0 = t0

    @property
    def timepoints(self):
        _timepoints = [self.t0]
        for period in self.periods:
            _timepoints.append(_timepoints[-1] + period.duration)
        return _timepoints

    @property
    def condition_ids(self):
        return [period.condition_id for period in self.periods]

    @staticmethod
    def from_timecourses(
        timecourses: Sequence['Timecourse'],
        durations: Sequence[float],
        *args,
        **kwargs,
    ) -> 'Timecourse':
        """Create a timecourse from a sequence of timecourses.

        Args:
            timecourses:
                The sequence of timecourses.
            durations:
                The duration of each timecourse except the last. Timecourses
                will be truncated if their total duration exceeds the duration
                specified here.
            *args, **kwargs:
                Passed to the `Timecourse.__init__` constructor.
        """
        if len(durations) != len(timecourses) - 1:
            raise ValueError(
                'Please specify one fewer durations than timecourses. The '
                'duration of the final timecourse will be unlimited.'
            )

        periods = []

        for timecourse_index, timecourse in enumerate(timecourses):
            # Simply add all periods from the last timecourse.
            if timecourse_index == len(durations):
                periods.extend(timecourse.periods)
                continue

            total_timecourse_duration = durations[timecourse_index]
            timecourse_duration = 0

            # Add only the periods that don't exceed the duration of the timecourse.
            for period in timecourse.periods:
                period_duration = period.duration

                # Truncate at period that exceeds the total duration.
                if period.duration + timecourse_duration > total_timecourse_duration:
                    period = copy.deepcopy(period)
                    period.duration = total_timecourse_duration - timecourse_duration
                    periods.append(period)
                    break

                timecourse_duration += period.duration
                periods.append(period)

        return Timecourse(
            periods=periods,
            *args,
            # FIXME change to argument like part of `last_measured_timepoints`?
            t0=timecourses[0].t0,
            **kwargs,
        )

    def to_df(self):
        data = {
            TIMECOURSE_ID: [self.timecourse_id],
            TIMECOURSE_NAME: [self.name],
            TIMECOURSE: [None],
        }
        if self.name is None:
            del data[TIMECOURSE_NAME]

        t0 = self.t0
        times = []
        condition_ids = []
        for period in self.periods:
            times.append(t0)
            condition_ids.append(period.condition_id)
            t0 += period.duration

        timecourse_str = PERIOD_DELIMITER.join(
            TIME_CONDITION_DELIMITER.join([str(time), condition_id])
            for time, condition_id in zip(times, condition_ids)
        )
        data[TIMECOURSE] = timecourse_str

        return get_timecourse_df(
            pd.DataFrame(data=data)
        )

    @staticmethod
    def from_df_row(row: pd.Series) -> 'Timecourse':
        periods = []
        period_sequence = [
            time__condition_id.split(TIME_CONDITION_DELIMITER)
            for time__condition_id in row.get(TIMECOURSE).split(PERIOD_DELIMITER)
        ]
        t0 = None
        for period_index, (start, condition_id) in enumerate(period_sequence):
            if t0 is None:
                try:
                    t0 = float(start)
                except ValueError:
                    t0 = start
            # Default to a period that is indefinite
            end = TIME_STEADY_STATE
            # End the period early if another period comes afterwards
            if period_index < len(period_sequence) - 1:
                end = period_sequence[period_index + 1][0]

            try:
                start = float(start)
                end = float(end)
            except ValueError:
                raise ValueError(
                    'Parameterized timepoints are not yet supported. '
                    'Please request.'
                )

            periods.append(
                Period(
                    duration=end-start,
                    condition_id=condition_id,
                )
            )

        return Timecourse(
            timecourse_id=row.name,
            name=row.get(TIMECOURSE_NAME),
            periods=periods,
            t0=t0,
        )

    @staticmethod
    def from_df(
        timecourse_df: pd.DataFrame,
        timecourse_id: str,
    ) -> 'Timecourse':
        return Timecourse.from_df_row(timecourse_df.loc[timecourse_id])

    def __len__(self):
        return len(self.periods)


def get_timecourse_df(
    timecourse_file: Union[str, pd.DataFrame, None]
) -> pd.DataFrame:
    """Read the provided condition file into a ``pandas.Dataframe``
    Conditions are rows, parameters are columns, conditionId is index.
    Arguments:
        condition_file: File name of PEtab condition file or pandas.Dataframe
    """
    if timecourse_file is None:
        return timecourse_file

    if isinstance(timecourse_file, (str, Path)):
        timecourse_file = pd.read_csv(
            timecourse_file,
            sep='\t',
            float_precision='round_trip',
        )

    petab.lint.assert_no_leading_trailing_whitespace(
        timecourse_file.columns.values, "timecourse"
    )

    if not isinstance(timecourse_file.index, pd.RangeIndex):
        timecourse_file.reset_index(inplace=True)

    try:
        timecourse_file.set_index([TIMECOURSE_ID], inplace=True)
    except KeyError:
        raise KeyError(
            f'Timecourse table missing mandatory field {TIMECOURSE_ID}.')

    return timecourse_file
