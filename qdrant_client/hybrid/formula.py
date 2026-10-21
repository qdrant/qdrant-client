import math
from datetime import datetime


from qdrant_client.conversions.common_types import get_args_subscribed
from qdrant_client.http import models
from typing import Any, Sequence

from qdrant_client.local import datetime_utils
from qdrant_client.local.geo import geo_distance
from qdrant_client.local.json_path_parser import parse_json_path
from qdrant_client.local.payload_filters import check_condition
from qdrant_client.local.payload_value_extractor import value_by_key

DEFAULT_SCORE = 0.0
DEFAULT_DECAY_TARGET = 0.0
DEFAULT_DECAY_MIDPOINT = 0.5
DEFAULT_DECAY_SCALE = 1.0


def evaluate_expression(
    expression: models.Expression,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    has_vector: dict[str, bool],
    defaults: dict[str, Any],
) -> float:
    if isinstance(expression, (float, int)):  # Constant
        return float(expression)

    elif isinstance(expression, str):  # Variable
        return evaluate_variable(expression, point_id, scores, payload, defaults)

    elif isinstance(expression, get_args_subscribed(models.Condition)):
        if check_condition(expression, payload, point_id, has_vector):  # type: ignore
            return 1.0
        return 0.0

    elif isinstance(expression, models.MultExpression):
        factors: list[float] = []

        for expr in expression.mult:
            factor = evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
            # Return early if any factor is zero
            if factor == 0.0:
                return factor

            factors.append(factor)

        return math.prod(factors)

    elif isinstance(expression, models.SumExpression):
        return sum(
            evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
            for expr in expression.sum
        )

    elif isinstance(expression, models.MaxExpression):
        # folds like core: no shortcuts, NaN operands are skipped
        maximum = -math.inf
        for expr in expression.max:
            value = evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
            maximum = max(maximum, value)
        return maximum

    elif isinstance(expression, models.MinExpression):
        minimum = math.inf
        for expr in expression.min:
            value = evaluate_expression(expr, point_id, scores, payload, has_vector, defaults)
            minimum = min(minimum, value)
        return minimum

    elif isinstance(expression, models.NegExpression):
        value = evaluate_expression(
            expression.neg, point_id, scores, payload, has_vector, defaults
        )
        return -value

    elif isinstance(expression, models.AbsExpression):
        return abs(
            evaluate_expression(expression.abs, point_id, scores, payload, has_vector, defaults)
        )

    elif isinstance(expression, models.DivExpression):
        left = evaluate_expression(
            expression.div.left, point_id, scores, payload, has_vector, defaults
        )

        if left == 0.0:
            return left

        right = evaluate_expression(
            expression.div.right, point_id, scores, payload, has_vector, defaults
        )

        if right == 0.0:
            if expression.div.by_zero_default is not None:
                return expression.div.by_zero_default
            raise_non_finite_error(f"{left}/{right}")

        result = left / right
        if math.isfinite(result):
            return result

        raise_non_finite_error(f"{left}/{right}")

    elif isinstance(expression, models.SqrtExpression):
        value = evaluate_expression(
            expression.sqrt, point_id, scores, payload, has_vector, defaults
        )

        if value >= 0:
            return math.sqrt(value)

        raise_non_finite_error(f"√{value}")

    elif isinstance(expression, models.PowExpression):
        base = evaluate_expression(
            expression.pow.base, point_id, scores, payload, has_vector, defaults
        )
        exponent = evaluate_expression(
            expression.pow.exponent, point_id, scores, payload, has_vector, defaults
        )

        try:
            # OverflowError: the result, or a too large integer operand, doesn't fit in a float.
            # ValueError: undefined, e.g. a negative base with a non-integer exponent.
            result = math.pow(base, exponent)
        except (OverflowError, ValueError):
            pass
        else:
            if math.isfinite(result):
                return result

        raise_non_finite_error(f"{base}^{exponent}")

    elif isinstance(expression, models.ExpExpression):
        value = evaluate_expression(
            expression.exp, point_id, scores, payload, has_vector, defaults
        )

        try:
            return math.exp(value)
        except OverflowError:
            raise_non_finite_error(f"exp({value})")

    elif isinstance(expression, models.Log10Expression):
        value = evaluate_expression(
            expression.log10, point_id, scores, payload, has_vector, defaults
        )

        if value > 0:
            try:
                return math.log10(value)
            except OverflowError:
                pass

        raise_non_finite_error(f"log10({value})")

    elif isinstance(expression, models.LnExpression):
        value = evaluate_expression(expression.ln, point_id, scores, payload, has_vector, defaults)

        if value > 0:
            try:
                return math.log(value)
            except OverflowError:
                pass

        raise_non_finite_error(f"ln({value})")

    elif isinstance(expression, models.AcoshExpression):
        value = evaluate_expression(
            expression.acosh, point_id, scores, payload, has_vector, defaults
        )

        if value >= 1:
            try:
                result = math.acosh(value)
            except OverflowError:
                pass
            else:
                if math.isfinite(result):
                    return result

        raise_non_finite_error(f"acosh({value})")

    elif isinstance(expression, models.GeoDistance):
        origin = expression.geo_distance.origin
        to = expression.geo_distance.to

        # Get value from payload
        geo_value = try_extract_payload_value(to, payload, defaults)

        if isinstance(geo_value, dict):
            destination = models.GeoPoint(**{key: geo_value[key] for key in ("lon", "lat") if key in geo_value})
            return geo_distance(origin.lon, origin.lat, destination.lon, destination.lat)

        raise ValueError(
            f"Expected geo point for {to} in the payload and/or in the formula defaults."
        )

    elif isinstance(expression, models.DatetimeExpression):
        return parse_datetime_constant(expression.datetime).timestamp()

    elif isinstance(expression, models.DatetimeKeyExpression):
        dt_str = try_extract_payload_value(expression.datetime_key, payload, defaults)
        dt = datetime_utils.parse(dt_str)
        if dt is None:
            raise ValueError(
                f"Expected datetime for {expression.datetime_key} in the payload and/or in the formula defaults."
            )

        return dt.timestamp()

    elif isinstance(expression, models.LinDecayExpression):
        x, target, midpoint, scale = evaluate_decay_params(
            expression.lin_decay, point_id, scores, payload, has_vector, defaults
        )

        lambda_factor = (1.0 - midpoint) / scale
        diff = abs(x - target)
        return max(0.0, -lambda_factor * diff + 1.0)

    elif isinstance(expression, models.ExpDecayExpression):
        x, target, midpoint, scale = evaluate_decay_params(
            expression.exp_decay, point_id, scores, payload, has_vector, defaults
        )

        lambda_factor = math.log(midpoint) / scale
        diff = abs(x - target)
        return math.exp(lambda_factor * diff)

    elif isinstance(expression, models.GaussDecayExpression):
        x, target, midpoint, scale = evaluate_decay_params(
            expression.gauss_decay, point_id, scores, payload, has_vector, defaults
        )

        lambda_factor = math.log(midpoint) / (scale * scale)
        diff = x - target
        return math.exp(lambda_factor * diff * diff)

    raise ValueError(f"Unsupported expression type: {type(expression)}")


def validate_expression(expression: models.Expression) -> None:
    """Reject what core refuses while parsing the formula, before any point is scored"""
    nested: Sequence[models.Expression | None]

    if isinstance(expression, str):
        variable = parse_variable(expression)
        if isinstance(variable, str):
            parse_json_path(variable)
        nested = []
    elif isinstance(expression, models.DatetimeExpression):
        parse_datetime_constant(expression.datetime)
        nested = []
    elif isinstance(expression, models.DatetimeKeyExpression):
        parse_json_path(expression.datetime_key)
        nested = []
    elif isinstance(expression, models.GeoDistance):
        parse_json_path(expression.geo_distance.to)
        nested = []
    elif isinstance(expression, models.MaxExpression):
        if len(expression.max) == 0:
            raise_no_operands_error("max")
        nested = expression.max
    elif isinstance(expression, models.MinExpression):
        if len(expression.min) == 0:
            raise_no_operands_error("min")
        nested = expression.min
    elif isinstance(expression, models.MultExpression):
        nested = expression.mult
    elif isinstance(expression, models.SumExpression):
        nested = expression.sum
    elif isinstance(expression, models.NegExpression):
        nested = [expression.neg]
    elif isinstance(expression, models.AbsExpression):
        nested = [expression.abs]
    elif isinstance(expression, models.DivExpression):
        nested = [expression.div.left, expression.div.right]
    elif isinstance(expression, models.SqrtExpression):
        nested = [expression.sqrt]
    elif isinstance(expression, models.PowExpression):
        nested = [expression.pow.base, expression.pow.exponent]
    elif isinstance(expression, models.ExpExpression):
        nested = [expression.exp]
    elif isinstance(expression, models.Log10Expression):
        nested = [expression.log10]
    elif isinstance(expression, models.LnExpression):
        nested = [expression.ln]
    elif isinstance(expression, models.AcoshExpression):
        nested = [expression.acosh]
    elif isinstance(expression, models.LinDecayExpression):
        validate_decay_params(expression.lin_decay, linear=True)
        nested = [expression.lin_decay.x, expression.lin_decay.target]
    elif isinstance(expression, models.ExpDecayExpression):
        validate_decay_params(expression.exp_decay, linear=False)
        nested = [expression.exp_decay.x, expression.exp_decay.target]
    elif isinstance(expression, models.GaussDecayExpression):
        validate_decay_params(expression.gauss_decay, linear=False)
        nested = [expression.gauss_decay.x, expression.gauss_decay.target]
    else:
        nested = []

    for nested_expression in nested:
        if nested_expression is not None:  # decay target is optional
            validate_expression(nested_expression)


def validate_decay_params(params: models.DecayParamsExpression, linear: bool) -> None:
    midpoint = params.midpoint if params.midpoint is not None else DEFAULT_DECAY_MIDPOINT
    if linear:
        if not 0.0 <= midpoint <= 1.0:
            raise ValueError(
                f"Linear decay midpoint should be in the range [0.0, 1.0], got {midpoint}"
            )
    elif midpoint <= 0.0 or midpoint >= 1.0:
        raise ValueError(f"Decay midpoint should be in the range (0.0, 1.0), got {midpoint}")

    scale = params.scale if params.scale is not None else DEFAULT_DECAY_SCALE
    if scale <= 0.0:
        raise ValueError(f"Scale must be non-zero positive, got {scale}")


def parse_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    # core parses the keys like variables, so `$score` and `$score[0]` share a default
    parsed = {}
    for key, value in defaults.items():
        variable = parse_variable(key)
        parsed[score_default_key(variable) if isinstance(variable, int) else key] = value
    return parsed


def score_default_key(index: int) -> str:
    return f"$score[{index}]"


def parse_datetime_constant(value: str) -> datetime:
    dt = datetime_utils.parse(value)
    if dt is None:
        raise ValueError(f"Expected datetime in supported format for {value}")
    return dt


def evaluate_decay_params(
    params: models.DecayParamsExpression,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    has_vector: dict[str, bool],
    defaults: dict[str, Any],
) -> tuple[float, float, float, float]:
    x = evaluate_expression(params.x, point_id, scores, payload, has_vector, defaults)

    if params.target is None:
        target = DEFAULT_DECAY_TARGET
    else:
        target = evaluate_expression(
            params.target, point_id, scores, payload, has_vector, defaults
        )

    midpoint = params.midpoint if params.midpoint is not None else DEFAULT_DECAY_MIDPOINT
    scale = params.scale if params.scale is not None else DEFAULT_DECAY_SCALE

    return x, target, midpoint, scale


def try_extract_payload_value(key: str, payload: models.Payload, defaults: dict[str, Any]) -> Any:
    # Get value from payload
    value = value_by_key(payload, key)

    if value is None or len(value) == 0:
        # Or from defaults
        value = defaults.get(key, None)
        # Consider it None if it is an empty list
        if isinstance(value, list) and len(value) == 0:
            value = None

    # Consider it a single value if it's a list with one element
    if isinstance(value, list) and len(value) == 1:
        return value[0]

    if value is None:
        raise ValueError(f"No value found for {key} in the payload nor the formula defaults")

    return value


def evaluate_variable(
    variable: str,
    point_id: models.ExtendedPointId,
    scores: list[dict[models.ExtendedPointId, float]],
    payload: models.Payload,
    defaults: dict[str, Any],
) -> float:
    var = parse_variable(variable)
    if isinstance(var, str):
        value = try_extract_payload_value(var, payload, defaults)

        if is_number(value):
            return value

        raise ValueError(
            f"Expected number value for {var} in the payload and/or in the formula defaults. Error: Value is not a number"
        )

    elif isinstance(var, int):
        # Get score from scores
        score = None
        if var < len(scores):
            score = scores[var].get(point_id, None)
            if score is not None:
                return score

        defined_default = defaults.get(score_default_key(var), None)
        if defined_default is not None:
            return defined_default

        return DEFAULT_SCORE

    raise ValueError(f"Invalid variable type: {type(var)}")


def parse_variable(var: str) -> str | int:
    # Try to parse score pattern
    if not var.startswith("$score"):
        # Treat as payload path
        return var

    remaining = var.replace("$score", "", 1)
    if remaining == "":
        # end of string, default idx is 0
        return 0

    # it must proceed with brackets
    if not remaining.startswith("["):
        raise ValueError(f"Invalid score pattern: {var}")

    remaining = remaining.replace("[", "", 1)
    bracket_end = remaining.find("]")
    if bracket_end == -1:
        raise ValueError(f"Invalid score pattern: {var}")

    # parse the content in between brackets as an unsigned integer. qdrant core represents
    # the score index as a usize, while `int()` would also accept a sign, underscore
    # separators and surrounding whitespace
    raw_idx = remaining[:bracket_end]
    if not (raw_idx.isascii() and raw_idx.isdigit()):
        raise ValueError(f"Invalid score pattern: {var}")

    idx = int(raw_idx)

    # make sure the string ends after the closing bracket
    if len(remaining) > bracket_end + 1:
        raise ValueError(f"Invalid score pattern: {var}")

    return idx


def raise_non_finite_error(expression: str) -> None:
    raise ValueError(f"The expression {expression} produced a non-finite number")


def raise_no_operands_error(operator: str) -> None:
    raise ValueError(f"`{operator}` needs at least one operand")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
