from dataclasses import dataclass


@dataclass(frozen=True)
class PersonId:
    """Identifier of a Person."""

    value: str


@dataclass(frozen=True)
class Person:
    """Person who takes part in Tasks."""

    id: PersonId
    name: str
