import pytest

from air_agent import (
    AirAgentError,
    ConfigurationError,
    MCPConnectionError,
    MemoryError,
    PlannerError,
    PluginLoadError,
    ProviderError,
    ToolExecutionError,
    ToolPermissionError,
)
from air_agent.tools.builtin import PermissionDeniedError


def test_public_error_hierarchy_is_exported():
    assert issubclass(ConfigurationError, AirAgentError)
    assert issubclass(ProviderError, AirAgentError)
    assert issubclass(ToolExecutionError, AirAgentError)
    assert issubclass(ToolPermissionError, ToolExecutionError)
    assert issubclass(ToolPermissionError, PermissionError)
    assert issubclass(MCPConnectionError, AirAgentError)
    assert issubclass(PluginLoadError, AirAgentError)
    assert issubclass(MemoryError, AirAgentError)
    assert issubclass(PlannerError, AirAgentError)


def test_permission_denied_error_remains_backward_compatible():
    assert issubclass(PermissionDeniedError, ToolPermissionError)
    with pytest.raises(PermissionError):
        raise PermissionDeniedError("denied")
