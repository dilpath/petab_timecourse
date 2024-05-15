from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import libsbml
import pandas as pd
import petab
from more_itertools import one
from slugify import slugify

#from .timecourse import Timecourse
from .petab import Condition
from .C import (
    TYPE_PATH,
    TIMECOURSE,
)
from .misc import parse_timecourse_string
from .timecourse import Timecourse


def remove_rules(variable: str, sbml_model: libsbml.Model):
    while sbml_model.removeRuleByVariable(variable):
        continue



def add_event(
        sbml_model: libsbml.Model,
        event_id: str,
        trigger_formula: str,
        event_assignments: Dict[str, Union[float, int, str]],
):
    """Add an event to an SBML model instance.

    The model is modified in-place.

    Parameters
    ----------
    sbml_model:
        The SBML model instance.
    event_id:
        The ID of the event.
    trigger_formula:
        The formula that describes when the event is triggered.
    event_assignments:
        A dictionary of assignments that occur when the event is triggered.
        Each key is an assignment target, and the corresponding value is value
        that the assignment target takes when the event is triggered.
    """
    event = sbml_model.createEvent()
    event.setId(event_id)
    event.setUseValuesFromTriggerTime(True)

    trigger = event.createTrigger()
    trigger.setInitialValue(True)
    trigger.setPersistent(True)
    trigger_math = libsbml.parseL3Formula(trigger_formula)
    trigger.setMath(trigger_math)

    for variable, event_assignment_formula in event_assignments.items():
        remove_rules(
            variable=variable,
            sbml_model=sbml_model,
        )
        event_assignment = event.createEventAssignment()
        event_assignment.setVariable(variable)
        event_assignment_math = \
            libsbml.parseL3Formula(str(event_assignment_formula))
        event_assignment.setMath(event_assignment_math)

        if variable in [p.getId() for p in sbml_model.getListOfParameters()]:
            set_parameter_as_not_constant(
                sbml_model=sbml_model,
                parameter_id=variable,
            )

def set_parameter_as_not_constant(
    sbml_model: libsbml.Model,
    parameter_id: str,
):
    """
    NB: changes the SBML model object inplace.
    """
    parameter = sbml_model.getParameter(parameter_id)
    parameter.setConstant(False)


def set_condition_parameters_not_constant(
    petab_problem: petab.Problem,
):
    """
    NB: changes the SBML model object inplace.
    """
    all_parameters = [
        parameter.getId()
        for parameter in petab_problem.sbml_model.getListOfParameters()
    ]
    parameter_ids = []
    for id in petab_problem.condition_df.columns:
        if id in all_parameters:
            parameter_ids.append(id)
    for parameter_id in parameter_ids:
        parameter = petab_problem.sbml_model.getParameter(parameter_id)
        parameter.setConstant(False)


def set_timecourse_parameters_not_constant(
    petab_problem: petab.Problem,
    timecourse: Timecourse,
):
    """
    NB: changes the SBML model object inplace.
    """
    warnings.warn('Setting all condition parameters not constant.')
    set_condition_parameters_not_constant(petab_problem=petab_problem)


def add_timecourse_as_events(
    petab_problem: petab.Problem,
    timecourse_id: str = None,
    output_path: Optional[TYPE_PATH] = None,
):
    if timecourse_id is None:
        try:
            timecourse_id = one(petab_problem.timecourse_df.index)
        except ValueError:
            raise ValueError(
                'A timecourse ID must be specified if there are multiple '
                'timecourses in the PEtab problem timecourse table.'
            )

    sbml_model = petab_problem.model.sbml_document.getModel()

    timecourse = Timecourse.from_df(
        timecourse_df=petab_problem.timecourse_df,
        timecourse_id=timecourse_id,
        condition_df=petab_problem.condition_df,
    )
    #timecourse = parse_timecourse_string(
    #    petab_problem.timecourse_df.loc[timecourse_id][TIMECOURSE],
    #)

    for timepoint, condition_id in zip(timecourse.timepoints, timecourse.condition_ids):
        add_event(
            sbml_model=sbml_model,
            event_id=get_slug(timepoint),
            trigger_formula=f'time >= {timepoint}',
            event_assignments=Condition(
                petab_problem.condition_df.loc[condition_id],
            ),
        )

    if output_path is not None:
        libsbml.writeSBMLToFile(petab_problem.sbml_document, str(output_path))

    return sbml_model


def get_slug(
        input_: Any,
        *args,
        **kwargs,
):
    """Get a human-readable string representation of the input.

    A description of what a slug is available at Wikipedia:
    https://en.wikipedia.org/wiki/Clean_URL#Slug

    The package `python-slugify` is used to generate the slug:
    https://github.com/un33k/python-slugify

    Parameters
    ----------
    input:
        The slug will be generated to represent this input.
    *args:
        Passed on to the `slugify` call (from the `python-slugify` package).
    **kwargs:
        Passed on to the `slugify` call (from the `python-slugify` package).
    """
    return slugify(str(input_), *args, separator='_', **kwargs)
