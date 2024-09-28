import abc
import copy
from typing import Any, Dict, List, Sequence, Tuple, Union

import amici.petab
#import amici.petab_import
from amici.petab import (
    RDATAS,
    EDATAS,
    LLH,
    SLLH,
)
from more_itertools import one
import numpy as np
import petab.v1
import petab.v2
from petab.C import (
    PARAMETER_ID,
)
#from pypesto.objective.amici_util import create_identity_parameter_mapping

from .C import (
    PERIODS,
)
from .fresh_amici import (
    precreate_edata_periods,
    precreate_parameter_mapping_periods,
    add_output_timepoints_if_missing,
)
#from .misc import (
#    #get_timecourse,
#    #subset_petab_problem,
#)
from .petab import (
    rescale_state_sensitivities,
)
#from .timecourse import Timecourse

AMICI = 'amici'
DATA = 'data'
SLLH_SUM = SLLH + '_sum'


class Simulator(abc.ABC):
    """Generic base class to simulate a PEtab Experiment.

    NB: currently probably not suitable for estimation problems
        where the time points at which a timecourse occurs is estimated.
        would need to recompute things like data in each petab problem
    """
    def __init__(
        self,
        petab_problem: petab.v2.Problem,
        experiment_ids: list[str] = None,
    ):
        self.petab_problem0 = petab_problem
        for experiment_id in self.petab_problem0.experiments.experiments:
            self.petab_problem0.experiments.experiments[experiment_id] = self.petab_problem0.experiments.denest(experiment_id)

        self.experiment_ids = experiment_ids
        if self.experiment_ids is None:
            self.experiment_ids = list(self.petab_problem0.experiments.experiments)

        self.experiments_petab_problems = petab.v2.experiments.get_v1_problem_sequence(
            petab_problem=self.petab_problem0,
            experiment_ids=self.experiment_ids,
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
        """Simulate an experiment.

        Args:
            problem_parameters_periods:
                Parameters to simulate, for each experiment period.

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
        """Simulate a single period of the experiment.

        Args:
            index:
                The index of the period in the experiment.
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
        #amici_model: amici.Model = None,
        #amici_solver: amici.Solver = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # FIXME add preequilibration and non-t0=0 support

        # represent parameters as a new value at each time period or
        # e.g. just the different values that the parameters take when they
        # change, and the times they change?

        # TODO custom model/solver setters/getters

        # FIXME set `edata.plist` to only compute derivatives
        # for relevant parameters in each period
        # FIXME set pscale here to all linear?
        self.experiments_amici_models = {}
        self.experiments_amici_solvers = {}
        self.experiments_amici_edatas = {}
        self.experiments_parameter_mappings_base = {}
        for experiment_id, experiment_petab_problems in self.experiments_petab_problems.items():
            print('setting up amici for..', end=" ")
            print(experiment_id)
            # make models
            self.experiments_amici_models[experiment_id] = [
                amici.petab.import_petab_problem(
                    experiment_petab_problem,
                    non_estimated_parameters_as_constants=True,
                )
                for i, experiment_petab_problem in enumerate(experiment_petab_problems.values())
                #if i % 2 == 1
            ]
            # make solvers
            self.experiments_amici_solvers[experiment_id] = [
                experiment_amici_model.getSolver()
                for experiment_amici_model in self.experiments_amici_models[experiment_id]
            ]
            # make edatas
            self.experiments_amici_edatas[experiment_id] = precreate_edata_periods(
                amici_models=self.experiments_amici_models[experiment_id],
                petab_problems=list(experiment_petab_problems.values()),
            )
            # make parameter mappings
            self.experiments_parameter_mappings_base[experiment_id] = precreate_parameter_mapping_periods(
                amici_models=self.experiments_amici_models[experiment_id],
                petab_problem0=self.petab_problem0,
                petab_problems=list(experiment_petab_problems.values()),
                experiment_id=experiment_id,
            )
        self.reset_parameter_mapping()
        self.experiment_ids = np.array(list(self.experiments_petab_problems))
        self.rng = np.random.default_rng(0)

    def set_amici_solver(self, settings: dict[str, Any]):
        for experiment_amici_solvers in self.experiments_amici_solvers.values():
            for experiment_amici_solver in experiment_amici_solvers:
                for setter, value in settings.items():
                    getattr(experiment_amici_solver, setter)(value)

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
        for _, parameter_mapping_periods in self.experiments_parameter_mapping_periods.items():
            for parameter_mapping_period in parameter_mapping_periods:
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
        self.experiments_parameter_mapping_periods = copy.deepcopy(self.experiments_parameter_mappings_base)

    #def problem_parameters_to_vector(
    #    self,
    #    problem_parameters: Dict[str, float],
    #) -> List[float]:
    #    return [
    #        problem_parameters[parameter_id]
    #        for parameter_id in self.amici_model.getParameterIds()
    #    ]

    #def shape_state_sensitivities(
    #    self,
    #    state_sensitivities: Sequence[float],
    #):
    #    return np.array(state_sensitivities).reshape(
    #        # TODO possibly need to use `nplist` for some
    #        #      cases... will probably produce an
    #        #      error when required
    #        self.amici_model.np(),
    #        self.amici_model.nx_rdata,
    #    )

    def simulate_experiments(
        self,
        experiment_ids: list[str] = None,
        percent: float = None,
        average_sllh: bool = True,
        **kwargs,
    ):
        """Simulate all experiments.

        Args:
            experiment_ids:
                The experiments that will be simulated. Defaults to all
                experiments.
            percent:
                The percent of experiments that can be simulated. Can be used
                to implement mini-batching. Defaults to all experiments
                (`100`%).
            average_sllh:
                Whether to average sensitivities across all experiments. If
                `False`, the sum is returned instead.
            kwargs:
                Passed to `AmiciSimulator.simulate`.

        Returns:
            Keys are experiment IDs, values are a list of simulated period results,
            one result per period.
        """
        if experiment_ids is None:
            experiment_ids = self.experiment_ids

        if percent is not None:
            experiment_ids = sorted(self.rng.choice(
                experiment_ids,
                size=int((percent/100)*experiment_ids.size),
                replace=False,
                shuffle=False,
            ))

        amici_results = {}
        data = {LLH: 0, SLLH_SUM: {}}
        for experiment_id in experiment_ids:
            #print(experiment_id)
            print(".", end="")
            result = self.simulate(experiment=self.petab_problem0.experiments[experiment_id], **kwargs)
            amici_results[experiment_id] = result[AMICI]
            data[LLH] += result[DATA][LLH]
            for parameter_id, sllh in result[DATA][SLLH_SUM].items():
                data[SLLH_SUM][parameter_id] = data[SLLH_SUM].get(parameter_id, 0) + sllh
        data[SLLH] = data[SLLH_SUM]
        if average_sllh:
            data[SLLH] = {
                k: v/len(experiment_ids)
                for k, v in data[SLLH].items()
            }
        return {AMICI: amici_results, DATA: data}

    def simulate(
        self,
        experiment: "Experiment",
        problem_parameters: dict[str, float] = None,
        problem_parameters_periods: list[list[str, float]] = None,
        scaled_parameters: bool = False,
    ):
        """Simulate a timecourse.

        Args:
            experiment:
                The experiment to simulate.
            problem_parameters_periods:
                Parameters to simulate, for each timecourse period.
            scaled_parameters:
                See `Simulator.simulate_period`.
                Whether the problem parameters are on their
                parameter scales (`True`) or on linear scale.

        Returns:
            TODO
        """
        if problem_parameters_periods is None:
            problem_parameters_periods = [copy.deepcopy(problem_parameters) for _ in self.experiments_petab_problems[experiment.experiment_id]]
            if problem_parameters_periods is None:
                problem_parameters_periods = [{} for _ in self.experiments_petab_problems[experiment.experiment_id]]

        x0 = ()
        sx0 = np.empty(0)
        #t0 = experiment.t0

        #plist = set()
        #for parameter_mapping_period in self.experiments_parameter_mapping_periods[experiment.experiment_id]:
        #    for plist_index, mapped_value in enumerate(one(parameter_mapping_period).map_sim_var.values()):
        #        if isinstance(mapped_value, str):
        #            plist.add(plist_index)
        #plist = sorted(plist)

        results = []
        data = {LLH: 0, SLLH_SUM: {}}
        for period_index, problem_parameters in enumerate(
            problem_parameters_periods,
        ):
            result = self.simulate_period(
                experiment=experiment,
                period_index=period_index,
                problem_parameters=problem_parameters,
                x0=x0,
                sx0=sx0.flatten(),
                #plist=plist,
                #t0=t0,
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
                if (
                    not (
                        end_state_previous_period
                        == start_state_current_period
                    ).all()
                    and not np.isnan(end_state_previous_period).all()
                    and not np.isnan(start_state_current_period).all()
                ):
                    import ipdb
                    ipdb.set_trace()
                    time_of_failure = list(self.experiments_petab_problems[experiment.experiment_id])[period_index]
                    raise NotImplementedError(
                        'In experiment `{experiment.experiment_id}`, '
                        'the end state of the previous period '
                        'and the start state of the current '
                        f'period (index: {period_index}) are not equal. '
                        'This could be due to e.g. an event that '
                        'occurs at the boundary between the two '
                        f'periods (timepoint: `{time_of_failure}`). '
                        'Such events are currently not supported, '
                        'as they may be triggered twice.'
                    )

            #t0 += experiment.get_period_duration(period_index)
            x0 = one(result[RDATAS]).x[-1].copy().flatten()
            # TODO default to e.g. `None` if sensis are not
            # requested by the user
            sx0 = one(result[RDATAS]).sx[-1].copy()

            results.append(result)

            if np.isnan(result[LLH]):
                import warnings
                warnings.warn("AMICI simulation failed. Setting LLH to inf.")
                data[LLH] += np.inf
                data[SLLH_SUM] = {}
            else:
                data[LLH] += result[LLH]
                for parameter_id, sllh in result[SLLH].items():
                    data[SLLH_SUM][parameter_id] = data[SLLH_SUM].get(parameter_id, 0) + sllh

        return {AMICI: results, DATA: data}

    def simulate_period(
        self,
        experiment: "Experiment",
        period_index: int,
        problem_parameters: Dict[str, float],
        x0: Sequence[float],
        sx0: Sequence[float],
        #plist: list[int],
        #t0: float,
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
        t0, petab_problem = list(self.experiments_petab_problems[experiment.experiment_id].items())[period_index]
        amici_edata = self.experiments_amici_edatas[experiment.experiment_id][period_index]
        parameter_mapping = self.experiments_parameter_mapping_periods[experiment.experiment_id][period_index]
        amici_model = self.experiments_amici_models[experiment.experiment_id][period_index]
        amici_solver = self.experiments_amici_solvers[experiment.experiment_id][period_index]


        period_times = list(self.experiments_petab_problems[experiment.experiment_id])
        try:
            period_duration = period_times[period_index + 1] - period_times[period_index]
        except IndexError:
            period_duration = np.inf
        #period_duration = experiment.get_period_duration(period_index)
        n_periods = len(period_times)

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
        #amici_edata.plist = plist

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
                    [t0 + period_duration]
                    if period_index < n_periods - 1
                    else []
                )
            ]
        )

        #if period_index == 15:
        #    import os
        #    os.environ['BREAK'] = "1"
        result = amici.petab.simulate_petab(
            petab_problem=petab_problem,
            amici_model=amici_model,
            solver=amici_solver,
            problem_parameters=problem_parameters,
            edatas=[amici_edata],
            parameter_mapping=parameter_mapping,
            scaled_parameters=scaled_parameters,
            #custom_plist=plist,
            scaled_gradients=True,
        )

        return result
