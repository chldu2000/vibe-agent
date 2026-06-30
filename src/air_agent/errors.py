"""Public exception hierarchy for Air Agent."""


class AirAgentError(Exception):
    """Base class for all public Air Agent errors."""


class ConfigurationError(AirAgentError, ValueError):
    """Raised when configuration is invalid."""


class ProviderError(AirAgentError, RuntimeError):
    """Raised when an LLM provider fails."""


class ToolExecutionError(AirAgentError, RuntimeError):
    """Raised when tool execution fails."""


class ToolPermissionError(ToolExecutionError, PermissionError):
    """Raised when a tool action is denied by permissions."""


class MCPConnectionError(AirAgentError, RuntimeError):
    """Raised when an MCP connection fails."""


class MCPToolError(ToolExecutionError):
    """Raised when an MCP server reports a tool-level failure."""


class PluginLoadError(AirAgentError, RuntimeError):
    """Raised when plugin loading fails."""


class MemoryError(AirAgentError, RuntimeError):
    """Raised when memory operations fail."""


class PlannerError(AirAgentError, ValueError):
    """Raised when planning fails."""


__all__ = [
    "AirAgentError",
    "ConfigurationError",
    "ProviderError",
    "ToolExecutionError",
    "ToolPermissionError",
    "MCPConnectionError",
    "MCPToolError",
    "PluginLoadError",
    "MemoryError",
    "PlannerError",
]
