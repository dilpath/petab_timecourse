from typing import Dict

#from .timecourse import Timecourse

#from .timecourse import Regimen  # FIXME: refactor to remove this dependency

condition_template  = '{value}, {start} <= time && time < {end}, '
condition_template2 = '{value}, and(leq({start}, time), lt(time, {end})), '

piecewise_template = 'piecewise({conditions}{default})'
# TODO was piece_template2
single_condition_piecewise_template = \
    piecewise_template.format(conditions=condition_template)


def regimen_as_single_piecewise(regimen: "Regimen") -> str:
    conditions = ''
    for administration in regimen._administrations:
        conditions += condition_template.format(  # TODO was condition_template2
            value=administration.value,
            start=administration.start,
            end=administration.end,
        )
    piecewise = piecewise_template.format(
        conditions=conditions,
        default=regimen.default,
    )
    return piecewise


def timecourse_as_single_piecewises(timecourse: "Timecourse") -> Dict[str, str]:
    raise NotImplementedError(
        'No longer in using "regimens". Will need to rewrite.'
    )
    return {
        regimen.target: regimen_as_single_piecewise(regimen)
        for regimen in timecourse.regimens
    }


def regimen_as_additive_piecewise(regimen: "Regimen") -> str:
    piecewises = []
    for index, administration in enumerate(regimen._administrations):
        # Only add default value to single additive piecewise.
        # TODO major issue?: currently assumes no overlapping conditions
        #                    in the regimen.
        if index == 0:
            value = administration.value
            default = regimen.default
        else:
            # TODO quite weird formulation at the moment, switch to
            #      multiplicative?
            value = administration.value - regimen.default,
            default = 0
        piecewises += single_condition_piecewise_template.format(
            start=administration.start,
            end=administration.end,
            value=value,
            default=default,
        )
    return ' + '.join(piecewises)


def timecourse_as_additive_piecewises(
        timecourse: "Timecourse",
) -> Dict[str, str]:
    raise NotImplementedError(
        'No longer in using "regimens". Will need to rewrite.'
    )
    return {
        regimen.target: regimen_as_additive_piecewise(regimen)
        for regimen in timecourse.regimens
    }
