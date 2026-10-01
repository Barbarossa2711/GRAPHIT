"""
Mastery model (PFA with time decay) and review due dates.

Two quantities serve two purposes (after Bjork & Bjork 1992): the time-dependent mastery
score (:func:`compute_mastery`, retrieval strength) only drives when a review is due and is
evaluated compute-on-read from the anchor ``t_last``; the monotonic peak ``mastery_peak``
(storage strength) only drives the PREREQUISITE gate and never decays. The adaptive
half-life at the end of this file is an outlook and is not wired into the system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

MASTERY_THRESHOLD = 0.8

GATE_THRESHOLD = 0.8


@dataclass(frozen=True)
class MasteryParams:
    """
    The three parameters of the PFA formula.

    ``t_h`` is the half-life in days, derived from the self-study budget of the module
    (6 ECTS -> 120 h, 60 % of it on the tutoring system -> about 3176 questions); 28 days
    exhausts that budget at an error rate of 18 %. For a module of different size it has to
    be recomputed from the module's ECTS budget. ``gamma`` (learning rate per
    correct answer) is matched to the item budget of the question generator (see
    :func:`s_min`), ``rho`` (learning rate per wrong answer, negative) follows from
    ``gamma/|rho| = 2.5``.
    """

    t_h: float = 28.0
    gamma: float = 0.75
    rho: float = -0.30


def today_daynum() -> int:
    """
    Returns today as a proleptic Gregorian day number, so differences are whole days.

    :return: Today's day number.
    """
    return date.today().toordinal()


def logit_threshold(threshold: float = MASTERY_THRESHOLD) -> float:
    """
    Computes the logit ``m*`` at which mastery reaches ``threshold`` (``ln(9)`` for 0.8).

    Derived from the threshold instead of hard-coded so that threshold and due formula
    cannot drift apart.

    :param threshold: Mastery value the logit is computed for.
    :return: The logit ``m*`` at which mastery reaches ``threshold``.
    """
    p = threshold / 2 + 0.5
    return math.log(p / (1 - p))


def compute_mastery(
    s: int,
    f: int,
    t_last: float,
    t_now: float,
    params: MasteryParams = MasteryParams(),
    *,
    t_h: float | None = None,
) -> float:
    """
    Computes mastery in [0, 1] at ``t_now`` (read-only, compute-on-read).

    Formula::

        λ       = ln(2) / t_h
        s_eff   = s · exp(−λ · (t_now − t_last))
        m       = γ · s_eff + ρ · f
        P       = 1 / (1 + exp(−m))
        mastery = max(0, (P − 0.5) · 2)

    :param s: Accumulated correct answers.
    :param f: Accumulated wrong answers.
    :param t_last: Day number of the last attempt.
    :param t_now: Day the value is evaluated for.
    :param params: Model parameters.
    :param t_h: Optional half-life overriding ``params``; only used by simulations.
    :return: Mastery in [0, 1] at ``t_now``.
    """
    t_h = params.t_h if t_h is None else t_h
    lam = math.log(2) / t_h
    s_eff = s * math.exp(-lam * (t_now - t_last))
    m = params.gamma * s_eff + params.rho * f
    p = 1 / (1 + math.exp(-m))
    return max(0.0, (p - 0.5) * 2)


def s_min(f: int = 0, params: MasteryParams = MasteryParams()) -> int:
    """
    Computes the minimum number of correct answers needed to reach the threshold at all.

    ``s_min = ceil((m* − ρ·f) / γ)``, evaluated right after an attempt where ``s_eff = s``.
    With ``γ = 0.75`` that is 3 correct answers, 4 after two wrong ones; hence the lower
    bound of 2 stems in the question generator.

    :param f: Number of wrong answers already accumulated.
    :param params: Model parameters supplying learning rate and threshold.
    :return: Correct answers required for the threshold to be reachable at all.
    """
    return math.ceil((logit_threshold() - params.rho * f) / params.gamma)


def due_interval(
    s: int, f: int, params: MasteryParams = MasteryParams(), *, t_h: float | None = None
) -> float | None:
    """
    Computes the continuous number of days from ``t_last`` until mastery drops below the threshold.

    Closed form of ``mastery ≥ MASTERY_THRESHOLD`` solved for ``Δt``::

        Δt_due = t_h / ln(2) · ln( γ·s / (m* − ρ·f) )

    Basis for evaluations and simulations; the date shown in the frontend comes from
    :func:`due_days`.

    :param s: Accumulated correct answers.
    :param f: Accumulated wrong answers.
    :param params: Model parameters supplying decay and threshold.
    :param t_h: Optional half-life overriding ``params``.
    :return: Days from ``t_last`` until mastery falls below the threshold; ``None`` if ``s``
        is too small to reach the threshold, meaning due from ``t_last`` on.
    """
    t_h = params.t_h if t_h is None else t_h
    counter = params.gamma * s
    denominator = logit_threshold() - params.rho * f
    if counter <= denominator:
        return None
    return t_h / math.log(2) * math.log(counter / denominator)


def due_days(
    s: int, f: int, params: MasteryParams = MasteryParams(), *, t_h: float | None = None
) -> int | None:
    """
    Quantises :func:`due_interval` to the whole day on which the concept actually becomes due.

    Day numbers are integers, so the status can only flip between calendar days. Uses
    ``floor(interval) + 1``: rounding misses the flip day in about 57 % of cases, and ``+1``
    instead of ``ceil`` covers an exactly integral interval, where the threshold still holds
    on that day because of ``>=``.

    :param s: Accumulated correct answers.
    :param f: Accumulated wrong answers.
    :param params: Model parameters supplying decay and threshold.
    :param t_h: Optional half-life; passed through to :func:`due_interval`.
    :return: Whole days from ``t_last`` until the status flips; ``None`` if the threshold
        is not reachable.
    """
    intervall = due_interval(s, f, params, t_h=t_h)
    return None if intervall is None else math.floor(intervall) + 1


def days_until_due(
    s: int,
    f: int,
    t_last: float,
    t_now: float,
    params: MasteryParams = MasteryParams(),
) -> int:
    """
    Computes the whole days left until a review is due; negative means overdue.

    If :func:`due_days` yields no date, the threshold is already missed at ``Δt = 0``, so the
    concept is due from ``t_last`` on and the elapsed days are returned negated. Whether the
    concept was ever mastered is not decided here but by ``mastery_peak`` via
    :func:`~Multiagent.LearnerModel.progress.gate_passed`.

    :param s: Accumulated correct answers.
    :param f: Accumulated wrong answers.
    :param t_last: Day number of the last attempt.
    :param t_now: Day the value is evaluated for.
    :param params: Model parameters supplying decay and threshold.
    :return: Whole days left until review is due; ``0`` means due today, negative overdue.
    """
    due_on = due_days(s, f, params)
    if due_on is None:
        due_on = 0
    return due_on - int(t_now - t_last)


@dataclass(frozen=True)
class AdaptiveParams:
    """
    Parameters of the adaptive half-life; outlook only, not used by the system.

    ``a`` is the maximum stability gain per success (at R -> 0), ``b`` the post-lapse
    factor applied to ``t_h`` after an error, ``t_h_max`` the upper bound of ``t_h``.
    """

    a: float = 1.5
    b: float = 0.5
    t_h_max: float = 365.0


def retrievability(t_h: float, dt: float) -> float:
    """
    Computes the retrieval strength ``R = 2^(−Δt/t_h)`` at the time of the question.

    :param t_h: Half-life of the memory trace, in days.
    :param dt: Days elapsed since the last attempt.
    :return: Retrieval strength in (0, 1]; close to 1 means easy recall.
    """
    lam = math.log(2) / t_h
    return math.exp(-lam * dt)


def update_on_answer(
    s: int,
    f: int,
    t_h: float,
    t_last: float,
    t_now: float,
    correct: bool,
    params: MasteryParams = MasteryParams(),
    adaptive: AdaptiveParams = AdaptiveParams(),
) -> tuple[int, int, float, float]:
    """
    Updates ``(s, f, t_h, t_last)`` after an answer with an adaptive half-life (outlook only).

    ``R`` must be computed with the old ``t_h`` and the old gap before ``t_h`` is updated.
    Correct: ``t_h ← min(t_h_max, t_h · (1 + a·(1 − R)))``; wrong:
    ``t_h ← max(params.t_h, b · t_h)``.

    :param s: Correct answers so far.
    :param f: Wrong answers so far.
    :param t_h: Current half-life of the memory trace.
    :param t_last: Day number of the previous attempt.
    :param t_now: Day of the answer being recorded.
    :param correct: Whether the student answered correctly.
    :param params: Model parameters; ``params.t_h`` is the floor of the half-life.
    :param adaptive: Parameters controlling how the half-life adapts.
    :return: Updated ``(s, f, t_h, t_last)``.
    """
    r = retrievability(t_h, t_now - t_last)
    if correct:
        s += 1
        t_h = min(adaptive.t_h_max, t_h * (1 + adaptive.a * (1 - r)))
    else:
        f += 1
        t_h = max(params.t_h, adaptive.b * t_h)
    return s, f, t_h, t_now


if __name__ == "__main__":
    p = MasteryParams()
    print(f"-- Implemented model (t_h={p.t_h}, gamma={p.gamma}, rho={p.rho}) --")
    print(f"s_min: f=0 -> {s_min(0)}   f=1 -> {s_min(1)}   f=2 -> {s_min(2)}")
    print("\nDue interval in days (per s correct and f wrong answers):")
    print(f"{'s':>3}{'f=0':>8}{'f=1':>8}{'f=2':>8}")
    for si in range(3, 9):
        row = f"{si:>3}"
        for fi in range(3):
            d = due_interval(si, fi)
            row += f"{(f'{d:.1f}' if d else '—'):>8}"
        print(row)

    def _simulate(intervals: list[int], label: str) -> None:
        """
        Prints the half-life reached after correct answers at the given intervals.

        :param intervals: Day intervals between the simulated answers.
        :param label: Caption printed in front of the result.
        """
        s = f = 0
        t_h = MasteryParams().t_h
        t = 0
        for dt in intervals:
            t += dt
            s, f, t_h, _ = update_on_answer(s, f, t_h, t - dt, t, correct=True)
        print(f"{label:26s} -> t_h = {t_h:6.1f} days   (s={s}, f={f})")

    print("\n-- OUTLOOK (not part of the system): spacing effect on t_h, 5 successes --")
    _simulate([1, 1, 1, 1, 1], "Massed (daily)")
    _simulate([7, 7, 7, 7, 7], "Weekly")
    _simulate([21, 21, 21, 21, 21], "Spaced (every 21 days)")
    print("-> Same number of successes, spaced practice builds more storage strength.")
