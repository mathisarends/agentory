import pytest

from agentory.mcp.schemas import (
    ClientInfo,
    InitializeParams,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPToolDefinition,
    MCPToolInputSchema,
    MCPToolsListResult,
)


class TestJsonRpcRequest:
    def test_default_jsonrpc_version(self) -> None:
        req = JsonRpcRequest(id=1, method="test")
        assert req.jsonrpc == "2.0"

    def test_params_default_to_empty(self) -> None:
        req = JsonRpcRequest(id=1, method="test")
        assert req.params == {}

    def test_serialization_roundtrip(self) -> None:
        req = JsonRpcRequest(id=42, method="tools/list", params={"key": "val"})
        data = req.model_dump()
        restored = JsonRpcRequest.model_validate(data)
        assert restored.id == 42
        assert restored.method == "tools/list"
        assert restored.params == {"key": "val"}


class TestJsonRpcResponse:
    def test_unwrap_returns_result(self) -> None:
        resp = JsonRpcResponse(jsonrpc="2.0", id=1, result={"ok": True})
        assert resp.unwrap() == {"ok": True}

    def test_unwrap_empty_result(self) -> None:
        resp = JsonRpcResponse(jsonrpc="2.0", id=1)
        assert resp.unwrap() == {}

    def test_unwrap_with_error_raises(self) -> None:
        resp = JsonRpcResponse(
            jsonrpc="2.0",
            id=1,
            error={"code": -1, "message": "boom"},
        )
        with pytest.raises(RuntimeError, match="MCP error"):
            resp.unwrap()


class TestJsonRpcNotification:
    def test_notification_no_params(self) -> None:
        n = JsonRpcNotification(method="notifications/initialized")
        assert n.params is None
        assert n.jsonrpc == "2.0"


class TestInitializeParams:
    def test_default_protocol_version(self) -> None:
        params = InitializeParams(clientInfo=ClientInfo(name="test", version="0.1"))
        assert params.protocolVersion == "2024-11-05"
        assert params.capabilities == {}


class TestMCPToolDefinition:
    def test_minimal_tool_definition(self) -> None:
        tool = MCPToolDefinition(name="read_file")
        assert tool.name == "read_file"
        assert tool.description is None
        assert tool.inputSchema.properties == {}
        assert tool.inputSchema.required == []

    def test_tool_with_schema(self) -> None:
        tool = MCPToolDefinition(
            name="search",
            description="Search files",
            inputSchema=MCPToolInputSchema(
                properties={"query": {"type": "string"}},
                required=["query"],
            ),
        )
        assert tool.description == "Search files"
        assert "query" in tool.inputSchema.properties


class TestMCPToolsListResult:
    def test_empty_tools_list(self) -> None:
        result = MCPToolsListResult()
        assert result.tools == []

    def test_parse_tools_list(self) -> None:
        result = MCPToolsListResult.model_validate(
            {
                "tools": [
                    {"name": "t1", "description": "Tool 1"},
                    {"name": "t2"},
                ]
            }
        )
        assert len(result.tools) == 2
        assert result.tools[0].name == "t1"
        assert result.tools[1].description is None
