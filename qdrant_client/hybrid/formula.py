from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from typing import Union, Any
import math

from qdrant_client.local.geo import geo_distance
from qdrant_client.local.payload_filters import check_condition
from qdrant_client.local.payload_value_extractor import value_by_key

DEFAULT_SCORE = 0.0

UNDEFINED_SCORE = float("-inf")


def evaluate_expression(
    expression: models.Expression,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    has_vector: dict[str, bool],
    defaults: dict[str, Any],
) -> float:
    if isinstance(expression, (models.StrictFloat, models.StrictInt)):  # Constant
        return expression
    elif isinstance(expression, models.StrictStr):  # Variable
        return evaluate_variable(expression, point_id, scores, payload, has_vector, defaults)

    elif isinstance(expression, models.Condition):
        if check_condition(expression, payload, point_id, has_vector):
            return 1.0
        return 0.0

    elif isinstance(expression, models.MultExpression):
        return math.prod(
            evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
            for expr in expression.mult
        )

    elif isinstance(expression, models.SumExpression):
        return math.fsum(
            evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
            for expr in expression.sum
        )

    elif isinstance(expression, models.NegExpression):
        return -evaluate_expression(
            expression.neg, point_id, scores, payload, has_vector, defaults
        )

    elif isinstance(expression, models.AbsExpression):
        return abs(
            evaluate_expression(expression.abs, point_id, scores, payload, has_vector, defaults)
        )

    elif isinstance(expression, models.DivExpression):
        left = evaluate_expression(
            expression.div.left, point_id, scores, payload, has_vector, defaults
        )

        if left == 0.0:
            if expression.div.by_zero_default:
                return expression.div.by_zero_default
            return float("inf")

        right = evaluate_expression(
            expression.right, point_id, scores, payload, has_vector, defaults
        )

        return left / right

    elif isinstance(expression, models.SqrtExpression):
        value = evaluate_expression(
            expression.sqrt, point_id, scores, payload, has_vector, defaults
        )

        if value < 0.0:
            return UNDEFINED_SCORE

        return math.sqrt(value)

    elif isinstance(expression, models.PowExpression):
        base = evaluate_expression(
            expression.pow.base, point_id, scores, payload, has_vector, defaults
        )
        exponent = evaluate_expression(
            expression.pow.exponent, point_id, scores, payload, has_vector, defaults
        )

        power = math.pow(base, exponent)

        if power == float("nan"):
            return UNDEFINED_SCORE

        return power

    elif isinstance(expression, models.ExpExpression):
        value = evaluate_expression(
            expression.exp, point_id, scores, payload, has_vector, defaults
        )

        return math.exp(value)
    elif isinstance(expression, models.Log10Expression):
        value = evaluate_expression(
            expression.log10, point_id, scores, payload, has_vector, defaults
        )

        if value <= 0.0:
            return UNDEFINED_SCORE

        return math.log(value)

    elif isinstance(expression, models.LnExpression):
        value = evaluate_expression(expression.ln, point_id, scores, payload, has_vector, defaults)

        if value <= 0.0:
            return UNDEFINED_SCORE

        return math.log(value)

    elif isinstance(expression, models.GeoDistance):
        origin = expression.geo_distance.origin
        to = expression.geo_distance.to

        # Get value from payload
        value = value_by_key(payload, to)
        if value is not None and len(value) > 0:
            value = value[0]
        else:
            value = defaults.get(to, None)
            if value is None:
                raise ValueError(f"Missing value for {to}")

        destination = construct(models.GeoPoint, **value)

        return geo_distance(origin.lon, origin.lat, destination.lon, destination.lat)

    else:
        raise ValueError(f"Unsupported expression type: {type(expression)}")


def evaluate_variable(
    variable: str,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    has_vector: dict[str, bool],
    defaults: dict[str, Any],
) -> float:
    var = parse_variable(variable)
    if isinstance(var, models.StrictStr):
        # Get value from payload
        value = value_by_key(payload, var)

        if value is not None and len(value) > 0:
            value = value[0]
            try:
                return float(value)
            except ValueError:
                # try to get from defaults
                pass

        defined_default = defaults.get(var, None)
        try:
            return float(defined_default)
        except ValueError:
            return DEFAULT_SCORE

    elif isinstance(var, models.StrictInt):
        # Get score from scores
        score = None
        if var < len(scores):
            score = scores[var].get(point_id, None)
            if score is not None:
                return score

        defined_default = defaults.get(variable, None)
        if defined_default is None:
            return DEFAULT_SCORE
        return defined_default

    raise ValueError(f"Invalid variable type: {type(var)}")


def parse_variable(var: str) -> Union[models.StrictStr, int]:
    # Try to parse score pattern
    if not var.startswith("$score"):
        # Treat as payload path
        return var

    remainder = var[6:]
    if remainder == "":
        # end of string, default idx is 0
        return 0

    # it must proceed with brackets
    if not remainder.startswith("["):
        raise ValueError(f"Invalid score pattern: {var}")

    remainder = remainder[1:]
    bracket_end = remainder.find("]")
    if bracket_end == -1:
        raise ValueError(f"Invalid score pattern: {var}")

    # try parsing the content in between brackets as integer
    try:
        idx = int(remainder[:bracket_end])
    except ValueError:
        raise ValueError(f"Invalid score pattern: {var}")

    # make sure the string ends after the closing bracket
    if len(remainder) > bracket_end + 1:
        raise ValueError(f"Invalid score pattern: {var}")

    return idx


def test_parsing_variable():
    assert parse_variable("$score") == 0
    assert parse_variable("$score[0]") == 0
    assert parse_variable("$score[1]") == 1
    assert parse_variable("$score[2]") == 2

    try:
        parse_variable("$score[invalid]")
        assert False
    except ValueError as e:
        assert str(e) == "Invalid score pattern: $score[invalid]"

    try:
        parse_variable("$score[10].other")
        assert False
    except ValueError as e:
        assert str(e) == "Invalid score pattern: $score[10].other"
