
from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.guardrails import validate_input, validate_output
from src.reasoning import inject_reasoning_instruction, extract_reasoning_and_response
from src.router import route


class AgentState(TypedDict, total=False):
    query: str
    profile: str
    route: str
    blocked: bool
    error_message: str
    messages: list[BaseMessage]
    rag_context: str
    final_response: str


def build_graph(llm_with_tools, vectorstore, tools_by_name, project_root):

    from pathlib import Path
    from src.rag import retrieve_context

    base = Path(project_root)

    # ---- Nós ----

    def guardrails_node(state: AgentState) -> dict:
        query = (state.get("query") or "").strip()
        ok, err = validate_input(query)
        if not ok:
            return {
                "blocked": True,
                "error_message": err or "Pergunta bloqueada.",
                "final_response": err or "Pergunta bloqueada.",
            }
        return {"blocked": False, "error_message": None}

    def router_node(state: AgentState) -> dict:
        query = state.get("query") or ""
        r = route(query)
        return {"route": r}

    def tool_chain_node(state: AgentState) -> dict:
        query = state.get("query") or ""
        prompt = f"""O usuário perguntou sobre gastos/despesas de um mês. Use as ferramentas consultar_gastos ou listar_meses_com_gastos.
- consultar_gastos exige o mês no formato 2026-01. "janeiro" → 2026-01; "fevereiro" → 2026-02; "março" → 2026-03 (ano 2026).
- Se não souber o mês, use listar_meses_com_gastos.

Pergunta: {query}"""
        prompt = inject_reasoning_instruction(prompt)
        return {"messages": [HumanMessage(content=prompt)]}

    def rag_chain_node(state: AgentState) -> dict:
        query = state.get("query") or ""
        docs = retrieve_context(vectorstore, query)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use o contexto das políticas abaixo para responder. Se precisar de dados de gastos por mês, use as ferramentas disponíveis.

Contexto (políticas):
{contexto}

Pergunta: {query}"""
        prompt = inject_reasoning_instruction(prompt)
        return {"rag_context": contexto, "messages": [HumanMessage(content=prompt)]}

    def direct_chain_node(state: AgentState) -> dict:
        query = state.get("query") or ""
        prompt = f"""Responda de forma breve e útil. Se for sobre gastos por mês, use as ferramentas consultar_gastos ou listar_meses_com_gastos.

Pergunta: {query}"""
        prompt = inject_reasoning_instruction(prompt)
        return {"messages": [HumanMessage(content=prompt)]}

    def llm_chain_node(state: AgentState) -> dict:
        messages = list(state.get("messages") or [])
        while True:
            resp = llm_with_tools.invoke(messages)
            if not getattr(resp, "tool_calls", None):
                return {"final_response": resp.content or ""}
            messages.append(resp)
            for tc in resp.tool_calls:
                fn = tools_by_name.get(tc["name"])
                out = fn.invoke(tc["args"]) if fn else "Ferramenta não encontrada."
                messages.append(
                    ToolMessage(content=str(out), tool_call_id=tc["id"])
                )

    def format_output_node(state: AgentState) -> dict:
        text = state.get("final_response") or ""
        reasoning, response = extract_reasoning_and_response(text)
        out = validate_output(response if response else text)
        return {"final_response": out}

    def blocked_node(state: AgentState) -> dict:
        return {"final_response": state.get("error_message") or "Pergunta bloqueada."}

    # ---- Roteamento ----

    def after_guardrails(state: AgentState) -> Literal["router", "blocked"]:
        return "blocked" if state.get("blocked") else "router"

    def after_router(state: AgentState) -> Literal["tool_chain", "rag_chain", "direct_chain"]:
        r = state.get("route") or "direct"
        if r == "tool":
            return "tool_chain"
        if r == "rag":
            return "rag_chain"
        return "direct_chain"

    # ---- Grafo ----

    builder = StateGraph(AgentState)

    builder.add_node("guardrails", guardrails_node)
    builder.add_node("router", router_node)
    builder.add_node("tool_chain", tool_chain_node)
    builder.add_node("rag_chain", rag_chain_node)
    builder.add_node("direct_chain", direct_chain_node)
    builder.add_node("llm_chain", llm_chain_node)
    builder.add_node("format_output", format_output_node)
    builder.add_node("blocked", blocked_node)

    builder.add_edge(START, "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        after_guardrails,
        {"router": "router", "blocked": "blocked"},
    )
    builder.add_conditional_edges(
        "router",
        after_router,
        {
            "tool_chain": "tool_chain",
            "rag_chain": "rag_chain",
            "direct_chain": "direct_chain",
        },
    )
    builder.add_edge("tool_chain", "llm_chain")
    builder.add_edge("rag_chain", "llm_chain")
    builder.add_edge("direct_chain", "llm_chain")
    builder.add_edge("llm_chain", "format_output")
    builder.add_edge("format_output", END)
    builder.add_edge("blocked", END)

    return builder.compile()
