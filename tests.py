import pytest

from marketplace import Employer, Applicant, handle_offer


@pytest.fixture
def variables():
    e_rule_follower = Employer("ERF", 0.5, 50, "Normal", rule_follower=True)
    e_rule_breaker = Employer("ERB", 0.5, 50, "Normal", rule_follower=False)
    e_liar = Employer("EL", 0.5, 50, "Normal", rule_follower=False, liar=True)

    a_rule_follower = Applicant("ARF", 0.5, 50, "Normal", rule_follower=True)
    a_rule_breaker = Applicant("ARB", 0.5, 50, "Normal", rule_follower=False)
    a_liar = Applicant("AL", 0.5, 50, "Normal", rule_follower=False, liar=True)

    yield e_rule_follower, e_rule_breaker, e_liar, a_rule_follower, a_rule_breaker, a_liar


def test_rejection(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables

    handle_offer(e_rule_follower, True, a_rule_follower, False, True)

    assert e_rule_follower.current_match is None
    assert a_rule_follower.current_match is None


def test_acceptance(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables

    handle_offer(e_rule_breaker, True, a_rule_breaker, True, True)

    assert e_rule_breaker.current_match is a_rule_breaker
    assert a_rule_breaker.current_match is e_rule_breaker


def test_acceptance_with_break(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables
    e_rule_breaker.current_match = a_rule_follower
    a_rule_breaker.current_match = e_rule_follower
    e_rule_follower.current_match = a_rule_breaker
    a_rule_follower.current_match = e_rule_breaker

    e_rule_breaker.preferences = [a_rule_breaker, a_rule_follower, a_liar]
    a_rule_breaker.preferences = [e_rule_breaker, e_rule_follower, e_liar]

    handle_offer(e_rule_breaker, True, a_rule_breaker, True, True)

    assert e_rule_breaker.current_match is a_rule_breaker
    assert a_rule_breaker.current_match is e_rule_breaker
    assert e_rule_follower.current_match is None
    assert a_rule_follower.current_match is None


def test_both_lying(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables
    e_liar.current_match = a_rule_breaker
    a_liar.current_match = e_rule_breaker
    e_rule_breaker.current_match = a_liar
    a_rule_breaker.current_match = e_liar

    e_rule_follower.preferences = [a_liar, a_rule_breaker, a_rule_follower]
    a_rule_follower.preferences = [e_liar, e_rule_breaker, e_rule_follower]
    e_liar.preferences = [a_rule_breaker, a_liar, a_rule_follower]
    a_liar.preferences = [e_liar, e_rule_breaker, e_rule_follower]

    # liarrs are currently matched with rule_breakers
    # employer prefers rule breaker to a_liar
    # applicant prefers liar to rule breaker
    handle_offer(e_liar, False, a_liar, True, False)

    assert e_liar.current_match is a_rule_breaker
    assert a_liar.current_match is e_liar
    assert e_rule_breaker.current_match is a_liar
    assert a_rule_breaker.current_match is e_liar


def test_applicant_lying(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables

    e_rule_breaker.preferences = [a_liar, a_rule_breaker, a_rule_follower]
    e_liar.preferences = [a_liar, a_rule_breaker, a_rule_follower]
    a_liar.preferences = [e_liar, e_rule_breaker, e_rule_follower]
    a_rule_breaker.preferences = [e_liar, e_rule_breaker, e_rule_follower]

    handle_offer(e_rule_breaker, True, a_liar, True, False)

    assert e_rule_breaker.current_match is a_liar
    assert a_liar.current_match is e_rule_breaker

    handle_offer(e_liar, False, a_liar, True, False)
    assert e_liar.current_match is a_liar
    assert e_rule_breaker.current_match is a_liar
    assert a_liar.current_match is e_liar


def test_employer_lying(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables
    a_rule_breaker.current_match = e_rule_breaker
    e_liar.current_match = a_liar
    a_liar.current_match = e_liar
    e_liar.preferences = [a_liar, a_rule_breaker, a_rule_follower]
    a_rule_breaker.preferences = [e_liar, e_rule_breaker, e_rule_follower]

    # employer liar is currently matched with a_liar
    # rule breaker is currentyl matched with rule breaker
    # employer liar offers to rule breaker

    handle_offer(e_liar, False, a_rule_breaker, True, True)

    assert e_liar.current_match is a_liar
    assert a_rule_breaker.current_match is e_liar
    assert e_rule_breaker.current_match is None
    assert a_liar.current_match is e_liar


def test_responding_to_offer(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables
    e_rule_breaker.current_match = a_rule_follower
    a_rule_breaker.current_match = e_rule_follower
    e_rule_follower.current_match = a_rule_breaker
    a_rule_follower.current_match = e_rule_breaker

    e_rule_breaker.preferences = [a_rule_breaker, a_rule_follower, a_liar]
    a_rule_breaker.preferences = [e_rule_breaker, e_rule_follower, e_liar]
    a_liar.preferences = [e_rule_breaker, e_rule_follower, e_liar]

    assert a_rule_breaker.respond_to_offer(e_rule_breaker) == (True, True)
    a_rule_breaker.current_match = e_rule_breaker
    assert a_rule_breaker.respond_to_offer(e_rule_follower) == (False, True)

    assert a_liar.respond_to_offer(e_rule_breaker) == (True, False)
    a_liar.current_match = e_rule_breaker
    assert a_liar.respond_to_offer(e_rule_breaker) == (True, False)


def test_offering(variables):
    (
        e_rule_follower,
        e_rule_breaker,
        e_liar,
        a_rule_follower,
        a_rule_breaker,
        a_liar,
    ) = variables

    e_rule_breaker.preferences = [a_rule_breaker, a_rule_follower, a_liar]
    e_liar.preferences = [a_rule_breaker, a_rule_follower, a_liar]
    a_rule_breaker.preferences = [e_rule_breaker, e_rule_follower, e_liar]

    e_rule_breaker.p_index = 0
    e_liar.p_index = 0

    assert e_rule_follower.make_offer() == (None, True)

    # rule breaker
    assert e_rule_breaker.make_offer() == (a_rule_breaker, True)
    assert e_rule_breaker.make_offer() == (a_rule_follower, True)
    e_rule_breaker.p_index = 1
    e_rule_breaker.current_match = a_rule_breaker
    assert e_rule_breaker.make_offer() == (None, True)

    # liar
    assert e_liar.make_offer() == (a_rule_breaker, False)
    e_liar.current_match = a_rule_breaker
    assert e_liar.make_offer() == (a_rule_follower, False)
