"""End-to-end tests for OpenTelemetry in Agentflow.

Covers real graph executions, span hierarchies, observability levels (SPANS, STANDARD, FULL),
tool execution tracing, agent LLM execution tracing, multi-agent handoffs, high concurrency stress tests,
error handling, interruption/resumption, streaming, and exporter integrations using the official
OpenTelemetry SDK InMemorySpanExporter.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentflow.core.graph import Agent, StateGraph, ToolNode
from agentflow.core.state import AgentState, Message
from agentflow.utils import Command
from agentflow.runtime.publisher import (
    CompositePublisher,
    ConsolePublisher,
    ObservabilityLevel,
    OtelPublisher,
    setup_langsmith,
    setup_logfire,
    setup_observability,
    setup_tracing,
)
from agentflow.runtime.publisher.events import (
    ContentType,
    Event,
    EventModel,
    EventType,
)
from agentflow.runtime.publisher.otel_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROMPT,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    GRAPH_MODEL,
    GRAPH_RUN_ID,
    GRAPH_THREAD_ID,
    GRAPH_TOTAL_STEPS,
    GRAPH_USER_ID,
    NODE_NAME,
    NODE_STEP,
    SESSION_ID,
    TOOL_NAME,
    TOOL_TYPE,
)
from agentflow.storage.checkpointer import InMemoryCheckpointer
from agentflow.utils.constants import END, START


# ── Test Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def otel_setup():
    """Create an isolated TracerProvider with an InMemorySpanExporter for testing."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("agentflow-test")
    return {"exporter": exporter, "provider": provider, "tracer": tracer}


# ── Node Functions for Testing ────────────────────────────────────────────────


def node_step_one(state: AgentState) -> AgentState:
    state.context.append(Message.text_message("Result from step 1", role="assistant"))
    return state


def node_step_two(state: AgentState) -> AgentState:
    state.context.append(Message.text_message("Result from step 2", role="assistant"))
    return state


def failing_node(state: AgentState) -> AgentState:
    raise RuntimeError("Intentional failure in node execution")


def sample_calc_tool(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def handoff_node_a(state: AgentState) -> Command:
    state.context.append(Message.text_message("Handing off to Node B", role="assistant"))
    return Command(goto="agent_b", update=state)


def handoff_node_b(state: AgentState) -> AgentState:
    state.context.append(Message.text_message("Completed in Node B", role="assistant"))
    return state


# ── E2E Test Suite ────────────────────────────────────────────────────


class TestOtelE2E:
    """End-to-end test cases verifying OpenTelemetry instrumentation in Agentflow."""

    @pytest.mark.asyncio
    async def test_full_graph_execution_span_hierarchy(self, otel_setup):
        """Verify complete multi-node graph execution produces proper span hierarchy and attributes."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.STANDARD)

        graph.add_node("step_1", node_step_one)
        graph.add_node("step_2", node_step_two)
        graph.set_entry_point("step_1")
        graph.add_edge("step_1", "step_2")
        graph.add_edge("step_2", END)

        compiled = graph.compile()

        config = {
            "thread_id": "thread-e2e-1",
            "run_id": "run-e2e-1",
            "user_id": "user-e2e-1",
        }
        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Start pipeline")]},
            config=config,
        )

        # Wait for all background publisher tasks to finish
        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        # 4 spans: __start__ node, step_1 node, step_2 node, and agentflow.graph root span
        assert len(spans) == 4, f"Expected 4 spans, got {[s.name for s in spans]}"

        graph_span = next(s for s in spans if s.name == "agentflow.graph")
        node_spans = [s for s in spans if s.name == "agentflow.node"]

        # Verify trace ID propagation (all spans share same trace_id)
        trace_id = graph_span.context.trace_id
        assert trace_id != 0
        for s in node_spans:
            assert s.context.trace_id == trace_id
            assert s.parent.span_id == graph_span.context.span_id, (
                f"Node span {s.attributes.get(NODE_NAME)} parent_span_id does not match graph span_id"
            )

        # Verify Graph Span Attributes
        assert graph_span.attributes[GEN_AI_SYSTEM] == "agentflow"
        assert graph_span.attributes[GEN_AI_OPERATION] == "graph"
        assert graph_span.attributes[GRAPH_THREAD_ID] == "thread-e2e-1"
        assert graph_span.attributes[GRAPH_RUN_ID] == "run-e2e-1"
        assert graph_span.attributes[GRAPH_USER_ID] == "user-e2e-1"
        assert graph_span.attributes[SESSION_ID] == "thread-e2e-1"
        assert graph_span.status.status_code in (StatusCode.UNSET, StatusCode.OK)

        # Verify Node Spans
        node_names = [s.attributes[NODE_NAME] for s in node_spans]
        assert "step_1" in node_names
        assert "step_2" in node_names
        for s in node_spans:
            assert s.attributes[GEN_AI_OPERATION] == "node"
            if s.attributes[NODE_NAME] != "__start__":
                assert NODE_STEP in s.attributes
            assert s.start_time <= s.end_time

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_tool_node_execution_spans(self, otel_setup):
        """Verify ToolNode execution emits tool spans with correct parentage and attributes."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.FULL)

        tool_node = ToolNode([sample_calc_tool])

        async def run_calc_node(state: AgentState, config: dict) -> AgentState:
            tool_res = await tool_node.invoke(
                name="sample_calc_tool",
                args={"a": 5, "b": 10},
                tool_call_id="call-calc-1",
                config=config,
                state=state,
            )
            state.context.append(tool_res)
            return state

        graph.add_node("calc_node", run_calc_node)
        graph.set_entry_point("calc_node")
        graph.add_edge("calc_node", END)

        compiled = graph.compile()
        config = {"thread_id": "t-tool", "run_id": "r-tool"}

        await compiled.ainvoke({"messages": [Message.text_message("calculate")]}, config=config)
        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == "agentflow.tool"]
        node_spans = [s for s in spans if s.name == "agentflow.node"]

        assert len(tool_spans) >= 1, f"Finished spans were: {[s.name for s in spans]}"
        tool_span = tool_spans[0]
        assert tool_span.attributes[GEN_AI_OPERATION] == "tool_call"
        assert tool_span.attributes[TOOL_NAME] == "sample_calc_tool"
        assert tool_span.attributes[TOOL_TYPE] == "local"

        # At FULL level, tool.input and tool.output events exist
        event_names = [e.name for e in tool_span.events]
        assert "tool.input" in event_names
        assert "tool.output" in event_names

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_multi_agent_handoff_tracing(self, otel_setup):
        """Verify multi-agent workflow with Command handoffs creates chained node spans under single graph trace."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.STANDARD)

        graph.add_node("agent_a", handoff_node_a)
        graph.add_node("agent_b", handoff_node_b)
        graph.set_entry_point("agent_a")
        graph.add_edge("agent_b", END)

        compiled = graph.compile()
        config = {"thread_id": "t-handoff-1", "run_id": "r-handoff-1"}

        result = await compiled.ainvoke(
            {"messages": [Message.text_message("Start handoff workflow")]},
            config=config,
        )
        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        # Spans: __start__, agent_a, agent_b, and agentflow.graph
        assert len(spans) == 4, f"Expected 4 spans, got {[s.name for s in spans]}"

        graph_span = next(s for s in spans if s.name == "agentflow.graph")
        node_spans = [s for s in spans if s.name == "agentflow.node"]
        node_names = [s.attributes[NODE_NAME] for s in node_spans]

        assert "agent_a" in node_names
        assert "agent_b" in node_names
        for s in node_spans:
            assert s.context.trace_id == graph_span.context.trace_id
            assert s.parent.span_id == graph_span.context.span_id

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_high_concurrency_stress_tracing(self, otel_setup):
        """Verify high-concurrency parallel executions through the same OtelPublisher maintain trace isolation."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.STANDARD)

        graph.add_node("step_1", node_step_one)
        graph.add_node("step_2", node_step_two)
        graph.set_entry_point("step_1")
        graph.add_edge("step_1", "step_2")
        graph.add_edge("step_2", END)

        compiled = graph.compile()

        # Run 10 parallel graph invocations concurrently
        concurrency_count = 10
        tasks = []
        for i in range(concurrency_count):
            cfg = {
                "thread_id": f"thread-stress-{i}",
                "run_id": f"run-stress-{i}",
                "user_id": f"user-stress-{i}",
            }
            tasks.append(
                compiled.ainvoke(
                    {"messages": [Message.text_message(f"Concurrent request {i}")]},
                    config=cfg,
                )
            )

        results = await asyncio.gather(*tasks)
        assert len(results) == concurrency_count

        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        # Each invocation produces 4 spans (1 graph + 3 nodes: __start__, step_1, step_2)
        expected_total_spans = concurrency_count * 4
        assert len(spans) == expected_total_spans, f"Expected {expected_total_spans} spans, got {len(spans)}"

        graph_spans = [s for s in spans if s.name == "agentflow.graph"]
        assert len(graph_spans) == concurrency_count

        # Verify all 10 graph executions have distinct trace IDs
        trace_ids = {s.context.trace_id for s in graph_spans}
        assert len(trace_ids) == concurrency_count, "Trace IDs must be unique per graph execution"

        # Verify all children belong strictly to their parent graph trace
        for g_span in graph_spans:
            run_id = g_span.attributes[GRAPH_RUN_ID]
            g_trace_id = g_span.context.trace_id
            g_span_id = g_span.context.span_id
            child_node_spans = [
                s for s in spans
                if s.name == "agentflow.node" and s.context.trace_id == g_trace_id
            ]
            assert len(child_node_spans) == 3, f"Run {run_id} must have exactly 3 child node spans"
            for c_span in child_node_spans:
                assert c_span.parent.span_id == g_span_id

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_observability_levels_spans_vs_standard_vs_full(self, otel_setup):
        """Test the 3 observability levels and verify what data is included/excluded in each."""
        # Level 1: SPANS (timing & structure only, no token usage / content)
        exp_spans = InMemorySpanExporter()
        prov_spans = TracerProvider()
        prov_spans.add_span_processor(SimpleSpanProcessor(exp_spans))
        tracer_spans = prov_spans.get_tracer("spans-test")

        pub_spans = OtelPublisher(tracer=tracer_spans, level=ObservabilityLevel.SPANS)
        run_id = "r-levels"

        start_evt = EventModel.default(
            {"run_id": run_id, "thread_id": "t1"},
            data={
                "model": "gpt-4o",
                "provider": "openai",
                "request_params": {"temperature": 0.7, "max_tokens": 100},
                "input_messages": [{"role": "user", "content": "secret prompt"}],
            },
            content_type=[ContentType.MESSAGE],
            event=Event.LLM_CALL,
            event_type=EventType.START,
            node_name="agent_node",
        )
        end_evt = EventModel.default(
            {"run_id": run_id, "thread_id": "t1"},
            data={
                "input_tokens": 50,
                "output_tokens": 20,
                "output_response": "secret response",
                "finish_reason": "stop",
                "response_id": "resp-123",
            },
            content_type=[ContentType.MESSAGE],
            event=Event.LLM_CALL,
            event_type=EventType.END,
            node_name="agent_node",
        )

        await pub_spans.publish(start_evt)
        await pub_spans.publish(end_evt)

        spans_level_spans = exp_spans.get_finished_spans()
        assert len(spans_level_spans) == 1
        s1 = spans_level_spans[0]
        # At SPANS level: no usage attributes, no request params, no content events
        assert GEN_AI_USAGE_INPUT_TOKENS not in s1.attributes
        assert GEN_AI_REQUEST_TEMPERATURE not in s1.attributes
        assert len(s1.events) == 0

        # Level 2: STANDARD (tokens, params included; content excluded)
        exp_std = InMemorySpanExporter()
        prov_std = TracerProvider()
        prov_std.add_span_processor(SimpleSpanProcessor(exp_std))
        tracer_std = prov_std.get_tracer("std-test")

        pub_std = OtelPublisher(tracer=tracer_std, level=ObservabilityLevel.STANDARD)
        await pub_std.publish(start_evt)
        await pub_std.publish(end_evt)

        std_spans = exp_std.get_finished_spans()
        assert len(std_spans) == 1
        s2 = std_spans[0]
        assert s2.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert s2.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert s2.attributes[GEN_AI_REQUEST_TEMPERATURE] == 0.7
        # Content NOT in attributes or events
        assert GEN_AI_INPUT_MESSAGES not in s2.attributes
        assert not any(e.name == "gen_ai.content.prompt" for e in s2.events)

        # Level 3: FULL (tokens, params, prompt/completion content all included)
        exp_full = InMemorySpanExporter()
        prov_full = TracerProvider()
        prov_full.add_span_processor(SimpleSpanProcessor(exp_full))
        tracer_full = prov_full.get_tracer("full-test")

        pub_full = OtelPublisher(tracer=tracer_full, level=ObservabilityLevel.FULL)
        await pub_full.publish(start_evt)
        await pub_full.publish(end_evt)

        full_spans = exp_full.get_finished_spans()
        assert len(full_spans) == 1
        s3 = full_spans[0]
        assert s3.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert s3.attributes[GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert GEN_AI_INPUT_MESSAGES in s3.attributes
        assert GEN_AI_OUTPUT_MESSAGES in s3.attributes
        event_names = [e.name for e in s3.events]
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names

    @pytest.mark.asyncio
    async def test_error_handling_and_status_codes(self, otel_setup):
        """Verify failed node and graph executions mark spans with StatusCode.ERROR and error events."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.STANDARD)

        graph.add_node("failing_node", failing_node)
        graph.set_entry_point("failing_node")
        graph.add_edge("failing_node", END)

        compiled = graph.compile()

        with pytest.raises(Exception):
            await compiled.ainvoke({"messages": [Message.text_message("trigger error")]})

        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        # 3 spans: __start__ (OK), failing_node (ERROR), graph (ERROR)
        assert len(spans) == 3, f"Expected 3 spans, got {[s.name for s in spans]}"

        failing_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
        assert len(failing_spans) >= 2  # failing_node + graph span

        for span in failing_spans:
            error_events = [e for e in span.events if "error" in e.name]
            assert len(error_events) >= 1

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_graph_interruption_and_resumption_tracing(self, otel_setup):
        """Verify interruption events and resumption events are recorded on graph spans."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]
        checkpointer = InMemoryCheckpointer()

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.STANDARD)

        graph.add_node("step_1", node_step_one)
        graph.add_node("step_2", node_step_two)
        graph.set_entry_point("step_1")
        graph.add_edge("step_1", "step_2")
        graph.add_edge("step_2", END)

        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["step_2"],
        )

        config = {"thread_id": "thread-interrupt-1", "run_id": "run-interrupt-1"}

        # 1. Run until interruption
        result = await compiled.ainvoke(
            {"messages": [Message.text_message("run until interrupt")]},
            config=config,
        )
        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        # Finished node spans during initial run: __start__ and step_1
        assert len(spans) >= 2

        # 2. Resume execution to completion
        resume_result = await compiled.ainvoke(
            {"messages": [Message.text_message("resume execution")]},
            config=config,
        )
        await compiled._task_manager.wait_for_all()

        finished_spans = exporter.get_finished_spans()
        # Graph finishes on resume -> graph span is finished
        assert any(s.name == "agentflow.graph" for s in finished_spans)
        graph_span = next(s for s in finished_spans if s.name == "agentflow.graph")
        graph_events = [e.name for e in graph_span.events]
        assert "graph.interrupted" in graph_events

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_streaming_execution_tracing(self, otel_setup):
        """Verify astream streaming execution produces spans correctly."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        graph = StateGraph()
        setup_tracing(graph, tracer=tracer, level=ObservabilityLevel.STANDARD)

        graph.add_node("step_1", node_step_one)
        graph.set_entry_point("step_1")
        graph.add_edge("step_1", END)

        compiled = graph.compile()
        config = {"thread_id": "stream-thread"}

        chunks = []
        async for chunk in compiled.astream(
            {"messages": [Message.text_message("Stream test")]},
            config=config,
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        # 3 spans: __start__, step_1, agentflow.graph
        assert len(spans) == 3, f"Expected 3 spans, got {[s.name for s in spans]}"
        assert any(s.name == "agentflow.graph" for s in spans)
        assert any(s.name == "agentflow.node" for s in spans)

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_composite_publisher_with_otel(self, otel_setup):
        """Verify OtelPublisher works properly within CompositePublisher when multiple publishers are attached."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        otel_pub = OtelPublisher(tracer=tracer, level=ObservabilityLevel.STANDARD)
        console_pub = ConsolePublisher()

        # Pass multiple publishers to StateGraph (automatically wrapped in CompositePublisher)
        graph = StateGraph(publisher=[otel_pub, console_pub])

        graph.add_node("step_1", node_step_one)
        graph.set_entry_point("step_1")
        graph.add_edge("step_1", END)

        compiled = graph.compile()
        await compiled.ainvoke({"messages": [Message.text_message("Multi publisher test")]})
        await compiled._task_manager.wait_for_all()

        spans = exporter.get_finished_spans()
        assert len(spans) == 3, f"Expected 3 spans, got {[s.name for s in spans]}"
        assert any(s.name == "agentflow.graph" for s in spans)

        await compiled.aclose()

    @pytest.mark.asyncio
    async def test_graceful_aclose_flushes_active_spans(self, otel_setup):
        """Verify aclose cleanly closes publisher and ends any lingering spans."""
        exporter: InMemorySpanExporter = otel_setup["exporter"]
        tracer: trace.Tracer = otel_setup["tracer"]

        pub = OtelPublisher(tracer=tracer)
        start_evt = EventModel.default(
            {"run_id": "run-lingering", "thread_id": "t1"},
            data={},
            content_type=[ContentType.STATE],
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
        )
        await pub.publish(start_evt)

        # Before close, 0 spans finished
        assert len(exporter.get_finished_spans()) == 0

        # After close, lingering span is ended
        await pub.close()
        assert pub._is_closed is True
        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        assert finished[0].name == "agentflow.graph"
