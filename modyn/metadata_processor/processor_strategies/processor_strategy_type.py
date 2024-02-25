"""Processor strategy type and exception."""

from enum import Enum


class ProcessorStrategyType(Enum):
    """Enum for the type of metadata processor strategy.

    Important: The value of the enum must be the same as the name of the module.
    The name of the enum must be the same as the name of the class.
    """

    BasicProcessorStrategy = "basic_processor_strategy"  # pylint: disable=invalid-name


class InvalidProcessorStrategyTypeException(Exception):
    """Exception for invalid processor strategy type."""

    def __init__(self, message: str):
        """Init exception.

        Args:
            message (str): Exception message
        """
        super().__init__(message)
