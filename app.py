
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.graph import build_graph
from src.rag import build_vectorstore
from src.tools import consultar_gastos, listar_meses_com_gastos

base = Path(__file__).resolve().parent
load_dotenv(base / ".env")
load_dotenv(base / ".env.local")

if not os.getenv("OPENAI_API_KEY"):
    print("Erro: OPENAI_API_KEY não definida")
    exit(1)

tools = [consultar_gastos, listar_meses_com_gastos]
tools_por_nome = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
llm_com_tools = llm.bind_tools(tools)
vectorstore = build_vectorstore(base)

graph = build_graph(llm_com_tools, vectorstore, tools_por_nome, base)

print("Assistente pronto. Pergunte sobre férias, reembolso, viagens ou gastos (ou digite sair).")

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break

    result = graph.invoke({"query": pergunta})
    print(f"\nAssistente: {result.get('final_response', '')}")
