import abc
import copy
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Sequence, Tuple, Union
import warnings

import amici
import amici.petab_import
from amici.petab_objective import (
    RDATAS,
)
from more_itertools import one
import numpy as np
import petab
from petab.C import (
    CONDITION_NAME,
    LIN,
    PARAMETER_ID,
)
#from pypesto.objective.amici_util import create_identity_parameter_mapping

from .C import (
    PERIODS,
)
from .amici import (
    precreate_edata_periods,
    precreate_parameter_mapping_periods,
    add_output_timepoints_if_missing,
)
from .misc import (
    get_timecourse,
    subset_petab_problem,
)
from .petab import (
    rescale_state_sensitivities,
)
from .timecourse import Timecourse


class Simulator(abc.ABC):
    """Generic base class to simulate a PEtab Timecourse.

    NB: currently probably not suitable for estimation problems
        where the time points at which a timecourse occurs is estimated.
        would need to recompute things like data in each petab problem
    """
    def __init__(
        self,
        petab_problem: petab.Problem,
        # TODO change to `timecourse: Timecourse = None`
        timecourse_id: str = None,
        t0: float = 0.0,
    ):
        self.petab_problem = petab_problem
        self.timecourse = Timecourse.from_df(
            timecourse_df=petab_problem.timecourse_df,
            timecourse_id=timecourse_id,
            condition_df=petab_problem.condition_df,
        )

        # FIXME t0 should not be 0 for preeq/presim
        self.t0 = t0

        # represent parameters as a new value at each time period or
        # e.g. just the different values that the parameters take when they
        # change, and the times they change?

        self.petab_problem_periods = subset_petab_problem(
            petab_problem=petab_problem,
            timecourse_id=timecourse_id,
        )

        # TODO create a dictionary that specifies the parameter values that
        #      must (?) be provided for each time period (i.e. the parameters)
        #      that get estimated.
        # self.required_parameters_list: List[List[str]]

    @abc.abstractmethod
    def simulate(
        self,
        problem_parameters_periods: List[Dict[str, float]] = None,
    ):
        """Simulate a timecourse.

        Args:
            problem_parameters_periods:
                Parameters to simulate, for each timecourse period.

        Returns:
            TODO standard return format?
        """
        raise NotImplementedError

    def simulate_period(
        self,
        index: int,
        problem_parameters: Dict[str, float],
        x0: Sequence[float],
        *args,
        **kwargs,
    ):
        """Simulate a single period of the timecourse.

        Args:
            index:
                The index of the timecourse period in the timecourse.
            problem_parameters:
                The parameters to simulate, on linear scale.
            x0:
                Initial state values.
            args, kwargs:
                Other additional information for the simulation.
                e.g. initial state sensitivities.

        Returns:
           TODO standard return format?
        """
        raise NotImplementedError


class AmiciSimulator(Simulator):
    """AMICI simulator for PEtab timecourses."""
    def __init__(
        self,
        amici_model: amici.Model = None,
        amici_solver: amici.Solver = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if float(self.t0) != 0:
            raise NotImplementedError(
                'Timecourses that do not start at t=0 are not yet supported. '
                'Please request. '
                f'The chosen timecourse `{timecourse_id}` starts at: '
                f'{self.t0}'
            )

        # represent parameters as a new value at each time period or
        # e.g. just the different values that the parameters take when they
        # change, and the times they change?

        # TODO custom model setters/getters
        self.amici_model = amici_model
        if self.amici_model is None:
            self.amici_model = amici.petab_import.import_petab_problem(
                self.petab_problem,
            )
        # TODO custom solver setters/getters
        self.amici_solver = amici_solver
        if self.amici_solver is None:
            self.amici_solver = self.amici_model.getSolver()
        # FIXME set `edata.plist` to only compute derivatives
        # for relevant parameters in each period
        # FIXME set pscale here to all linear
        self.amici_edata_periods = precreate_edata_periods(
            amici_model=self.amici_model,
            petab_problems=self.petab_problem_periods,
        )
        self.parameter_mapping_periods_base = precreate_parameter_mapping_periods(
            amici_model=self.amici_model,
            petab_problems=self.petab_problem_periods,
            timecourse_id=self.timecourse.timecourse_id,
        )
        self.reset_parameter_mapping()

    def replace_in_parameter_mapping(
        self,
        replacements: Dict[str, float],
        scaled: bool = False,
    ):
        """Replace parameters in the parameter mapping with specific values.

        For example, some species `species_x` may take the parameter `initial_species_x`
        as its initial value. This method can be used to replace `initial_species_x`
        with a specific value.

        Args:
            replacements:
                Keys are IDs, values are the values that will replace the IDs.
            scaled:
                Whether all values in `replacements` are on the scales defined in the
                parameter mapping. If not, values are assumed to be on linear scale will
                be scaled.
        """
        # Replace everywhere in the parameter mapping
        for parameter_mapping_period in self.parameter_mapping_periods:
            for parameter_mapping_for_condition in parameter_mapping_period:
                for mapping_attr in ['map_sim_var', 'map_preeq_fix', 'map_sim_fix']:
                    setattr(parameter_mapping_for_condition, mapping_attr, {
                        k: (
                            # If replaceable, replace
                            (
                                replacements[v]
                                if scaled
                                # Rescale replacement if necessary
                                else petab.parameters.scale(
                                    parameter=replacements[v],
                                    scale_str=getattr(
                                        parameter_mapping_for_condition,
                                        'scale_' + mapping_attr,
                                    )[k],
                                )
                            )
                            if v in replacements
                            # Else use current value
                            else v
                        )
                        for k, v in getattr(
                            parameter_mapping_for_condition,
                            mapping_attr,
                        ).items()
                    })


    def reset_parameter_mapping(self):
        """Reset to undo previous customizations of the parameter mapping."""
        self.parameter_mapping_periods = copy.deepcopy(self.parameter_mapping_periods_base)

    def problem_parameters_to_vector(
        self,
        problem_parameters: Dict[str, float],
    ) -> List[float]:
        return [
            problem_parameters[parameter_id]
            for parameter_id in self.amici_model.getParameterIds()
        ]

    def shape_state_sensitivities(
        self,
        state_sensitivities: Sequence[float],
    ):
        return np.array(state_sensitivities).reshape(
            # TODO possibly need to use `nplist` for some
            #      cases... will probably produce an
            #      error when required
            self.amici_model.np(),
            self.amici_model.nx_rdata,
        )

    def simulate(
        self,
        problem_parameters_periods: List[Dict[str, float]] = None,
        scaled_parameters: bool = False,
        control_parameters: Dict[str, Dict[str, Any]] = None,
    ):
        """Simulate a timecourse.

        Args:
            problem_parameters_periods:
                Parameters to simulate, for each timecourse period.
            scaled_parameters:
                See `Simulator.simulate_period`.
                Whether the problem parameters are on their
                parameter scales (`True`) or on linear scale.
            control_parameters:
                Provide information about how to reset sensitivities
                during the simulation, to correctly compute sensitivities
                for parameters that appear for a subset of periods
                in a timecourse. Keys are control parameter IDs,
                values are dictionaries with the following key, value
                pairs:
                    parameterId:
                        The ID of the model parameter that corresponds to
                        this control parameter.
                    periods:
                        A list of period indices, in ascending order,
                        in which this control parameter is active.
                        Different control parameters for the same
                        model parameter cannot be simultaneously active
                        in the same period.

        Returns:
            TODO
        """
        if problem_parameters_periods is None:
            problem_parameters_periods = [{} for _ in self.timecourse.periods]

        x0 = ()
        sx0 = np.empty(0)
        t0 = self.t0

        results = []
        for period_index, (period, problem_parameters) in enumerate(zip(
            self.timecourse.periods,
            problem_parameters_periods,
        )):
            # Continue sensitivities from previous simulations
            if (
                control_parameters is not None
                and
                period_index > 0
            ):
                for (
                    control_parameter_id,
                    control_description,
                ) in control_parameters.items():
                    if (
                        period_index
                        not in control_description[PERIODS]
                    ):
                        continue

                    control_specific_period_index = (
                        control_description[PERIODS]
                        .index(period_index)
                    )

                    # Parameter position in simulation list objects
                    parameter_id = control_description[PARAMETER_ID]
                    # FIXME rewrite so sensis for requested parameter
                    #       IDs is computed instead? Instead of all
                    #       estimated parameters?
                    parameter_index_model = (
                        self
                        .amici_model
                        .getParameterIds()
                        .index(parameter_id)
                    )
                    parameter_index_sensis = (
                        self.amici_edata_periods
                        [period_index-1]
                        .plist
                        .index(parameter_index_model)
                    )

                    # Start calculating sensitivity for a new
                    # parameter
                    if control_specific_period_index == 0:
                        next_sx0 = 0.0
                    # Or continue calculating sensitivity for an old
                    # parameter, by initializing at the sensitivities
                    # at the end of the previous period for this
                    # control parameter.
                    else:
                        previous_control_period_index = (
                            control_description
                            [PERIODS]
                            [control_specific_period_index - 1]
                        )
                        next_sx0 = (
                            one(
                                results
                                [previous_control_period_index]
                                [RDATAS]
                            )
                            .sx[-1]
                            [parameter_index_sensis]
                        )

                    # Changes it for all states
                    sx0[parameter_index_sensis] = next_sx0

            result = self.simulate_period(
                period_index=period_index,
                problem_parameters=problem_parameters,
                x0=x0,
                sx0=sx0.flatten(),
                t0=t0,
                scaled_parameters=scaled_parameters,
            )

            # Events etc. that change the state at the initial
            # time point are not yet supported, to avoid having
            # events trigger at the end of the previous period and
            # start of the next period.
            if period_index > 0:
                end_state_previous_period = x0
                start_state_current_period = \
                    one(result[RDATAS]).x[0].flatten()
                if not (
                    end_state_previous_period
                    == start_state_current_period
                ).all():
                    raise NotImplementedError(
                        'The end state of the previous period '
                        'and the start state of the current '
                        f'period (index: {period_index}) are not equal. '
                        'This could be due to e.g. an event that '
                        'occurs at the boundary between the two '
                        f'periods (timepoint: {t0}). '
                        'Such events are currently not supported, '
                        'as they may be triggered twice.'
                    )

            t0 += period.duration
            x0 = one(result[RDATAS]).x[-1].copy().flatten()
            # TODO default to e.g. `None` if sensis are not
            # requested by the user
            sx0 = one(result[RDATAS]).sx[-1].copy()

            results.append(result)

        return results

    def simulate_period(
        self,
        period_index: int,
        problem_parameters: Dict[str, float],
        x0: Sequence[float],
        sx0: Sequence[float],
        t0: float,
        scaled_parameters: bool = False,
    ):
        """Simulate a single period of the timecourse.

        Args:
            period_index:
                The index of the timecourse period in the timecourse.
            problem_parameters:
                The parameters to simulate.
            x0:
                Initial state values.
                Should correspond to `amici_model.getStateIds()`.
            sx0:
                Initial state sensitivities, on parameter scale.
            t0:
                The start time.
            scaled_parameters:
                Whether the problem parameters are on their
                PEtab (Control) parameter scales (`True`) or on
                linear scale (`False`).

        Returns:
           TODO
        """
        petab_problem = self.petab_problem_periods[period_index]
        amici_edata = self.amici_edata_periods[period_index]
        parameter_mapping = self.parameter_mapping_periods[period_index]

        period = self.timecourse.periods[period_index]

        if petab_problem.measurement_df.empty:
            # FIXME resolve this properly. Is it due to timecourse pieces
            #       that are defined to occur after the last measured time
            #       point?
            pass
        # NOTE removed replacement of control parameters with ID of time-period-specific
        #       control parameter

        if t0 < 0:
            raise NotImplementedError(
                'Presimulation/preequilibration is not yet implemented. Please request.'
            )

        # FIXME expects float -- doesn't support parameterized/estimated timepoints
        amici_edata.tstart_ = t0
        amici_edata.x0 = x0
        amici_edata.sx0 = sx0

        # Simulation output should be generated at the endpoint of the period, to obtain
        # initial states and state sensitivities for the next period.
        # This is irrelevant for the last period, and should be avoided as PEtab
        # Timecourse currently sets the end of the last period to the steady-state time,
        # which doesn't exist for all models.
        # FIXME check if this is safe
        #       i.e. does it affect the likelihood function?
        #           seems not, as `nan`s are automatically added to the data,
        #           for appended timepoints (measurement data are not duplicated)

        add_output_timepoints_if_missing(
            amici_edata=amici_edata,
            timepoints=[
                t0,
                *(
                    [t0 + period.duration]
                    if period_index < len(self.timecourse) - 1
                    else []
                )
            ]
        )

        #breakpoint()
        result = amici.petab_objective.simulate_petab(
            petab_problem=petab_problem,
            amici_model=self.amici_model,
            solver=self.amici_solver,
            problem_parameters=problem_parameters,
            edatas=[amici_edata],
            parameter_mapping=parameter_mapping,
            scaled_parameters=scaled_parameters,
        )

        return result
