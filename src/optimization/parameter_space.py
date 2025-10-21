"""Parameter space definitions and helpers for optimization searches."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

Validator = Callable[[dict[str, Any]], None]
_DECIMAL_TOLERANCE = Decimal("1e-12")


@dataclass(frozen=True)
class ParameterDefinition:
    """Immutable description of a single search parameter."""

    name: str
    description: str = ""
    values: Sequence[Any] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    step: float | None = None
    is_required: bool = True

    def is_discrete(self) -> bool:
        """Return True when discrete values are provided."""
        return self.values is not None

    def has_range(self) -> bool:
        """Return True when continuous bounds are configured."""
        return self.lower_bound is not None or self.upper_bound is not None

    def generate_candidates(self) -> tuple[Any, ...]:
        """
        Produce a tuple of candidate values suitable for grid/random search.

        Raises:
            ValueError: if neither explicit values nor a bounded range with step is defined.
        """
        if self.values is not None:
            candidates = tuple(self.values)
            if not candidates:
                raise ValueError(f"Parameter {self.name} must define at least one candidate value")
            return candidates

        if self.lower_bound is None or self.upper_bound is None or self.step is None:
            raise ValueError(
                f"Parameter {self.name} must define either discrete values or a bounded range with step"
            )
        if self.upper_bound < self.lower_bound:
            raise ValueError(f"Parameter {self.name} has upper_bound < lower_bound")

        step_decimal = Decimal(str(self.step))
        if step_decimal <= 0:
            raise ValueError(f"Parameter {self.name} requires a positive step size")

        lower = Decimal(str(self.lower_bound))
        upper = Decimal(str(self.upper_bound))

        values: list[Decimal] = []
        current = lower
        # Guard against pathological configurations that could loop forever.
        max_iterations = 1_000_000
        while current <= upper + _DECIMAL_TOLERANCE:
            values.append(current)
            current += step_decimal
            if len(values) > max_iterations:
                raise ValueError(
                    f"Parameter {self.name} produced more than {max_iterations} grid values. "
                    "Check range and step configuration."
                )

        if not values:
            raise ValueError(f"Parameter {self.name} did not yield any grid values")

        return tuple(_coerce_decimal(value) for value in values)


@dataclass
class ParameterSpace:
    """Container for parameter definitions used by optimization strategies."""

    parameters: MutableMapping[str, ParameterDefinition] = field(default_factory=dict)
    validators: list[Validator] = field(default_factory=list)

    @classmethod
    def from_definitions(cls, definitions: Sequence[ParameterDefinition]) -> "ParameterSpace":
        """Construct a parameter space from an ordered sequence of definitions."""
        parameters: Dict[str, ParameterDefinition] = {}
        for definition in definitions:
            if definition.name in parameters:
                raise ValueError(f"Duplicate parameter definition: {definition.name}")
            parameters[definition.name] = definition
        return cls(parameters=parameters)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        validators: Iterable[Validator] | None = None,
    ) -> "ParameterSpace":
        """Load a parameter space definition from a YAML/JSON style mapping."""
        parameters_config = config.get("parameters", {}) if config else {}
        definitions: list[ParameterDefinition] = []
        for name, raw_definition in parameters_config.items():
            definitions.append(
                ParameterDefinition(
                    name=name,
                    description=raw_definition.get("description", ""),
                    values=_as_tuple(raw_definition.get("values")),
                    lower_bound=raw_definition.get("lower_bound"),
                    upper_bound=raw_definition.get("upper_bound"),
                    step=raw_definition.get("step"),
                    is_required=raw_definition.get("required", True),
                )
            )
        space = cls.from_definitions(definitions)
        if validators:
            space.validators.extend(list(validators))
        return space

    def to_config(self) -> dict[str, Any]:
        """Serialize the parameter space back into a configuration mapping."""
        config: dict[str, Any] = {"parameters": {}}
        for name, definition in self.parameters.items():
            item: dict[str, Any] = {}
            if definition.description:
                item["description"] = definition.description
            if definition.values is not None:
                item["values"] = list(definition.values)
            if definition.lower_bound is not None:
                item["lower_bound"] = definition.lower_bound
            if definition.upper_bound is not None:
                item["upper_bound"] = definition.upper_bound
            if definition.step is not None:
                item["step"] = definition.step
            if not definition.is_required:
                item["required"] = False
            config["parameters"][name] = item
        return config

    def add_validator(self, validator: Validator) -> None:
        """Register an additional validator executed after built-in checks."""
        self.validators.append(validator)

    def validate(self, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalise a parameter mapping.

        Args:
            parameters: Arbitrary mapping with candidate parameter values.

        Returns:
            A cleaned dictionary containing validated parameter values.

        Raises:
            ValueError: When required parameters are missing, values fall outside the allowed domain,
                or unknown parameters are supplied.
        """
        validated: dict[str, Any] = {}
        provided_keys = set(parameters.keys())

        for name, definition in self.parameters.items():
            if name not in provided_keys:
                if definition.is_required:
                    raise ValueError(f"Missing required parameter: {name}")
                continue

            value = parameters[name]
            validated[name] = self._validate_value(definition, value)

        extra_keys = sorted(provided_keys - set(self.parameters))
        if extra_keys:
            raise ValueError(f"Unknown parameter(s): {', '.join(extra_keys)}")

        for validator in self.validators:
            validator(validated)

        return validated

    def grid(self, overrides: Mapping[str, Sequence[Any]] | None = None) -> Dict[str, tuple[Any, ...]]:
        """
        Build a dictionary of parameter -> candidate sequence for grid/random search.

        Args:
            overrides: Optional mapping with explicit candidate lists for selected parameters.

        Returns:
            Ordered dictionary with tuples of candidate values.
        """
        candidates: Dict[str, tuple[Any, ...]] = {}
        overrides = overrides or {}

        unexpected_overrides = set(overrides) - set(self.parameters)
        if unexpected_overrides:
            unexpected = ", ".join(sorted(unexpected_overrides))
            raise ValueError(f"Overrides provided for unknown parameters: {unexpected}")

        for name, definition in self.parameters.items():
            if name in overrides:
                override_values = tuple(overrides[name])
                if not override_values:
                    raise ValueError(f"Override for parameter {name} must contain at least one value")
                candidates[name] = override_values
                continue
            candidates[name] = definition.generate_candidates()
        return candidates

    def iter_grid(self, overrides: Mapping[str, Sequence[Any]] | None = None) -> Iterator[dict[str, Any]]:
        """Iterate over the full cartesian product of the parameter grid."""
        grid_definition = self.grid(overrides)
        if not grid_definition:
            return

        keys = list(grid_definition.keys())
        for combination in product(*(grid_definition[key] for key in keys)):
            candidate = dict(zip(keys, combination))
            yield self.validate(candidate)

    def _validate_value(self, definition: ParameterDefinition, value: Any) -> Any:
        """Validate a single parameter value against its definition."""
        if definition.is_discrete():
            assert definition.values is not None  # for type checkers
            if value not in definition.values:
                allowed = ", ".join(map(repr, definition.values))
                raise ValueError(f"Value {value!r} is not permitted for parameter {definition.name}. Allowed: {allowed}")
            return value

        if definition.lower_bound is None and definition.upper_bound is None:
            raise ValueError(f"Parameter {definition.name} has no validation rule for value {value!r}")

        if not isinstance(value, (int, float)):
            raise TypeError(f"Parameter {definition.name} expects numeric values, got {type(value).__name__}")

        lower = definition.lower_bound
        upper = definition.upper_bound
        if lower is not None and value < lower:
            raise ValueError(f"Value {value!r} is below lower bound {lower!r} for parameter {definition.name}")
        if upper is not None and value > upper:
            raise ValueError(f"Value {value!r} is above upper bound {upper!r} for parameter {definition.name}")

        if definition.step is not None and lower is not None:
            offset = (Decimal(str(value)) - Decimal(str(lower))) / Decimal(str(definition.step))
            if offset % 1 and not _is_close_to_integer(offset):
                raise ValueError(
                    f"Value {value!r} for parameter {definition.name} does not align with step {definition.step}"
                )
        return value


def _is_close_to_integer(value: Decimal) -> bool:
    """Return True when the decimal value is effectively an integer."""
    nearest = value.to_integral_value()
    return abs(value - nearest) <= _DECIMAL_TOLERANCE


def _coerce_decimal(value: Decimal) -> Any:
    """Convert Decimal back to int or float, preserving integer representations."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _as_tuple(values: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if values is None:
        return None
    return tuple(values)
