from typing import Sequence, Tuple

#from petab_timecourse import Timecourse
from petab import Experiment


translate_condition = {
    'q_positive': 1,
    'q_zero': 0,
    'q_negative': -1,
}


def get_analytical_x_(
        t: float,
) -> float:
    """The expected value for $x_$.

    Parameters
    ----------
    t:
        The current time.
    timecourse_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    timecourse_p_ = [
        (0, 0.5),
        (10, 2.0),
        (20, 1.0),
        (30, 1.5),
        (40, 2.0),
    ]
    timecourse_q_ = [
        (0, 1),
        (10, 0),
        (20, -1),
        (30, 0),
        (40, 1),
    ]
    x_ = 0
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        p_ = timecourse_p_[row_index][1]
        # FIXME remove None
        if q_ is None or row_index == len(timecourse_q_) - 1:
            x_ += p_ * q_ * (t - t_start)
            break
        t_end = timecourse_q_[row_index+1][0]
        if t <= t_end:
            x_ += p_ * q_ * (t - t_start)
            break
        x_ += p_ * q_ * (t_end - t_start)
    return x_


def get_analytical_sx_(
        t: float,
) -> float:
    """The expected sensitivity for x_ w.r.t. p_.

    Parameters
    ----------
    t:
        The current time.
    timecourse_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    timecourse_p_ = [
        (0, 0.5),
        (10, 2.0),
        (20, 1.0),
        (30, 1.5),
        (40, 2.0),
    ]
    timecourse_q_ = [
        (0, 1),
        (10, 0),
        (20, -1),
        (30, 0),
        (40, 1),
    ]
    sx_ = 0
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        # FIXME remove
        p_ = timecourse_p_[row_index][1]
        # FIXME remove None
        if q_ is None or row_index == len(timecourse_q_) - 1:
            sx_ += q_ * (t - t_start)
            break
        t_end = timecourse_q_[row_index+1][0]
        if t <= t_end:
            sx_ += q_ * (t - t_start)
            break
        sx_ += q_ * (t_end - t_start)
    return sx_
