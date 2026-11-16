"""Grade only the artifact against protected, independently checked reserves."""
import json
import math
from pathlib import Path
import pytest

OUTPUT = Path('/app/reserves.json')
REFERENCE = Path('/tests/ground_truth.json')
QUERY = Path('/tests/open_claims.json')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f'Duplicate JSON key: {key}'
        result[key] = value
    return result


def reject_constant(value):
    raise AssertionError(f'Non-finite JSON constant: {value}')


def read_output(path=OUTPUT):
    assert path.is_file() and not path.is_symlink(), 'Expected regular /app/reserves.json'
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


@pytest.fixture(scope='module')
def output():
    return read_output()


@pytest.fixture(scope='module')
def reference():
    return json.loads(REFERENCE.read_text(encoding='utf-8'))


def numeric(value, probability=False):
    assert type(value) in (int, float), 'Expected a number, not a boolean or string'
    assert math.isfinite(value) and value >= 0, 'Expected finite nonnegative value'
    if probability:
        assert value <= 1, 'Probability exceeds one'


def close_amount(actual, expected):
    assert abs(actual - expected) <= max(25, .01 * abs(expected)), (actual, expected)


def test_structure_and_ids(output):
    """The JSON has the disclosed keys and every query ID exactly once, sorted."""
    assert isinstance(output, dict) and set(output) == {'claims', 'portfolios'}
    query = json.loads(QUERY.read_text(encoding='utf-8'))
    for collection, id_field, fields in [
        ('claims', 'claim_id', {'claim_id', 'expected_unpaid_indemnity_usd', 'standard_deviation_usd'}),
        ('portfolios', 'portfolio_id', {'portfolio_id', 'expected_unpaid_indemnity_usd', 'standard_deviation_usd', 'probability_above_attachment'})]:
        rows = output[collection]
        assert isinstance(rows, list)
        expected_ids = sorted(row[id_field] for row in query[collection])
        assert len(rows) == len(expected_ids)
        for row in rows:
            assert isinstance(row, dict) and set(row) == fields
            assert isinstance(row[id_field], str)
        assert [row[id_field] for row in rows] == expected_ids


def test_numeric_contract(output):
    """All reported values are finite numbers with their disclosed bounds."""
    for row in output['claims']:
        numeric(row['expected_unpaid_indemnity_usd'])
        numeric(row['standard_deviation_usd'])
    for row in output['portfolios']:
        numeric(row['expected_unpaid_indemnity_usd'])
        numeric(row['standard_deviation_usd'])
        numeric(row['probability_above_attachment'], probability=True)


def test_claim_expectations(output, reference):
    """Claim means condition on historical and current calendar/event evidence."""
    for actual, expected in zip(output['claims'], reference['claims'], strict=True):
        assert actual['claim_id'] == expected['claim_id']
        close_amount(actual['expected_unpaid_indemnity_usd'], expected['expected_unpaid_indemnity_usd'])


def test_claim_uncertainty(output, reference):
    """Claim uncertainty retains both latent processes and policy-cap mass."""
    for actual, expected in zip(output['claims'], reference['claims'], strict=True):
        assert actual['claim_id'] == expected['claim_id']
        close_amount(actual['standard_deviation_usd'], expected['standard_deviation_usd'])


def test_portfolio_expectations(output, reference):
    """Pools include the future indemnity of their listed open claims only."""
    for actual, expected in zip(output['portfolios'], reference['portfolios'], strict=True):
        assert actual['portfolio_id'] == expected['portfolio_id']
        close_amount(actual['expected_unpaid_indemnity_usd'], expected['expected_unpaid_indemnity_usd'])


def test_portfolio_tail_probabilities(output, reference):
    """Attachment probabilities use the joint posterior predictive distribution."""
    for actual, expected in zip(output['portfolios'], reference['portfolios'], strict=True):
        assert actual['portfolio_id'] == expected['portfolio_id']
        assert abs(actual['probability_above_attachment'] - expected['probability_above_attachment']) <= .005, (actual, expected)


def test_portfolio_uncertainty(output, reference):
    """Pool uncertainty includes event, calendar and temporal covariance."""
    for actual, expected in zip(output['portfolios'], reference['portfolios'], strict=True):
        assert actual['portfolio_id'] == expected['portfolio_id']
        close_amount(actual['standard_deviation_usd'], expected['standard_deviation_usd'])
