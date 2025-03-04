import warnings
from qdrant_client._pydantic_compat import construct
from qdrant_client.http import models
from typing import Union, Any
import math
import numpy as np

from qdrant_client.local.geo import geo_distance
from qdrant_client.local.payload_filters import check_condition
from qdrant_client.local.payload_value_extractor import value_by_key

DEFAULT_SCORE = np.float32(0.0)

DEFAULT_BY_ZERO = np.float32(1.0)


def evaluate_expression(
    expression: models.Expression,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    has_vector: dict[str, bool],
    defaults: dict[str, Any],
) -> np.float32:
    if isinstance(expression, (float, int)):  # Constant
        return np.float32(expression)

    elif isinstance(expression, str):  # Variable
        return evaluate_variable(expression, point_id, scores, payload, has_vector, defaults)

    elif isinstance(expression, models.Condition):
        if check_condition(expression, payload, point_id, has_vector):
            return np.float32(1.0)
        return np.float32(0.0)

    elif isinstance(expression, models.MultExpression):
        result = np.prod(
            [
                evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
                for expr in expression.mult
            ],
            dtype=np.float32,
        )
        return np.float32(result)

    elif isinstance(expression, models.SumExpression):
        result = np.sum(
            [
                evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
                for expr in expression.sum
            ],
            dtype=np.float32,
        )
        return np.float32(result)

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

        if left == np.float32(0.0):
            return left

        right = evaluate_expression(
            expression.div.right, point_id, scores, payload, has_vector, defaults
        )

        if right == 0.0:
            if expression.div.by_zero_default is not None:
                return np.float32(expression.div.by_zero_default)
            raise_non_finite_error(f"{left}/{right}")

        with np.errstate(invalid="ignore"):
            return left / right

    elif isinstance(expression, models.SqrtExpression):
        value = evaluate_expression(
            expression.sqrt, point_id, scores, payload, has_vector, defaults
        )

        with np.errstate(invalid="ignore"):
            sqrt_value = np.sqrt(value, dtype=np.float32)
            if np.isfinite(sqrt_value):
                return np.float32(sqrt_value)

        raise_non_finite_error(f"√{value}")

    elif isinstance(expression, models.PowExpression):
        base = evaluate_expression(
            expression.pow.base, point_id, scores, payload, has_vector, defaults
        )
        exponent = evaluate_expression(
            expression.pow.exponent, point_id, scores, payload, has_vector, defaults
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            power = np.power(base, exponent, dtype=np.float32)
            if np.isfinite(power):
                return np.float32(power)

        raise_non_finite_error(f"{base}^{exponent}")

    elif isinstance(expression, models.ExpExpression):
        value = evaluate_expression(
            expression.exp, point_id, scores, payload, has_vector, defaults
        )
        return np.exp(value, dtype=np.float32)

    elif isinstance(expression, models.Log10Expression):
        value = evaluate_expression(
            expression.log10, point_id, scores, payload, has_vector, defaults
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            log_value = np.log10(value, dtype=np.float32)
            if np.isfinite(log_value):
                return log_value

        raise_non_finite_error(f"log10({value})")

    elif isinstance(expression, models.LnExpression):
        value = evaluate_expression(expression.ln, point_id, scores, payload, has_vector, defaults)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ln_value = np.log(value, dtype=np.float32)
            if np.isfinite(ln_value):
                return ln_value

        raise_non_finite_error(f"ln({value})")

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

        return np.float32(geo_distance(origin.lon, origin.lat, destination.lon, destination.lat))

    raise ValueError(f"Unsupported expression type: {type(expression)}")


def evaluate_variable(
    variable: str,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    has_vector: dict[str, bool],
    defaults: dict[str, Any],
) -> np.float32:
    var = parse_variable(variable)
    if isinstance(var, str):
        # Get value from payload
        value = value_by_key(payload, var)

        if value is not None and len(value) > 0:
            value = value[0]
            try:
                return np.float32(value)
            except (TypeError, ValueError):
                # try to get from defaults
                pass

        defined_default = defaults.get(var, None)

        print(f"defined_default: {defined_default}")
        try:
            return np.float32(defined_default)
        except ValueError:
            return DEFAULT_SCORE

    elif isinstance(var, int):
        # Get score from scores
        score = None
        if var < len(scores):
            score = scores[var].get(point_id, None)
            if score is not None:
                return np.float32(score)

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


def raise_non_finite_error(expression: str):
    raise ValueError(f"The expression {expression} produced a non-finite number")


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
