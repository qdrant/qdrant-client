import random

import qdrant_client.models as models
from tests.fixtures.filters import one_random_condition_please


def one_random_expression_please(current_depth: int = 0, max_depth: int = 5) -> models.Expression:
    """Generate a random expression for testing purposes.

    Args:
        current_depth: Current depth of expression nesting
        max_depth: Maximum allowed depth for nested expressions

    Returns:
        A random Expression object
    """

    def rand_decay_params():
        return models.DecayParamsExpression(
            x=one_random_expression_please(current_depth + 1, max_depth),
            target=one_random_expression_please(current_depth + 1, max_depth),
            midpoint=round(random.uniform(0.00001, 0.99999), 5),
            scale=round(random.uniform(-1_000_000.0, 1_000_000.0), 5),
        )

    # Choose a random expression type with possible nesting
    expression_choices = [
        # Terminal expressions (no nesting) with higher probability at deeper levels
        lambda: round(random.uniform(-10.0, 10.0), 5),  # Constant
        lambda: random.choice(  # Variable
            [
                "rand_number",
                "rand_digit",
                "rand_signed_int",
                "nested.rand_digit",
                "$score",
                "$score[0]",
                "mixed_type",
            ]
        ),
        lambda: one_random_condition_please(),  # Condition
        # Nested expressions
        lambda: models.MultExpression(
            mult=[
                one_random_expression_please(current_depth + 1, max_depth)
                for _ in range(random.randint(2, 3))
            ]
        ),
        lambda: models.SumExpression(
            sum=[
                one_random_expression_please(current_depth + 1, max_depth)
                for _ in range(random.randint(2, 3))
            ]
        ),
        lambda: models.NegExpression(
            neg=one_random_expression_please(current_depth + 1, max_depth)
        ),
        lambda: models.AbsExpression(
            abs=one_random_expression_please(current_depth + 1, max_depth)
        ),
        lambda: models.DivExpression(
            div=models.DivParams(
                left=one_random_expression_please(current_depth + 1, max_depth),
                right=one_random_expression_please(current_depth + 1, max_depth),
                by_zero_default=round(random.uniform(0, 1), 5) if random.random() < 0.5 else None,
            )
        ),
        lambda: models.SqrtExpression(
            sqrt=one_random_expression_please(current_depth + 1, max_depth)
        ),
        lambda: models.PowExpression(
            pow=models.PowParams(
                base=one_random_expression_please(current_depth + 1, max_depth),
                exponent=one_random_expression_please(current_depth + 1, max_depth),
            )
        ),
        lambda: models.ExpExpression(
            exp=one_random_expression_please(current_depth + 1, max_depth)
        ),
        lambda: models.Log10Expression(
            log10=one_random_expression_please(current_depth + 1, max_depth)
        ),
        lambda: models.LnExpression(ln=one_random_expression_please(current_depth + 1, max_depth)),
        # GeoDistance is a special case - needs specific structure
        lambda: models.GeoDistance(
            geo_distance=models.GeoDistanceParams(
                origin=models.GeoPoint(
                    lon=round(random.uniform(-180, 180), 5),
                    lat=round(random.uniform(-90, 90), 5),
                ),
                to="city.geo",  # Using a field that would contain geo coordinates
            )
        ),
        lambda: models.LinDecayExpression(lin_decay=rand_decay_params()),
        lambda: models.ExpDecayExpression(exp_decay=rand_decay_params()),
        lambda: models.GaussDecayExpression(gauss_decay=rand_decay_params()),
    ]
    # If we've reached max depth, return a terminal expression (no nesting)
    if current_depth >= max_depth:  # Limit nesting depth
        # Return a simple expression at max depth
        return random.choice(expression_choices[:3])()

    # Give higher weight to terminal expressions at deeper levels
    if current_depth > 2:
        # Add more terminal expressions to increase their probability
        expression_choices = expression_choices[:3] * 3 + expression_choices[3:]

    return random.choice(expression_choices)()
