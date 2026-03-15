import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from src.rag import build_vectorstore, retrieve_context
from src.tools import consultar_gastos, listar_meses_com_gastos
from src.guardrails import validate_input, validate_output

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

print("Assistente pronto. Pergunte sobre férias, reembolso, viagens ou gastos (ou digite sair).")

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break

    ok, erro = validate_input(pergunta)
    if not ok:
        print(f"\nAssistente: {erro}")
        continue

    docs = retrieve_context(vectorstore, pergunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Use o contexto das políticas abaixo para responder. Se a pergunta for sobre gastos/despesas por mês, use as ferramentas disponíveis para consultar os dados.

Regras:
- Responda com base no contexto quando for sobre políticas.
- Para perguntas sobre quanto gastamos, em qual mês, etc., use as tools consultar_gastos ou listar_meses_com_gastos.
- Só diga "Não encontrei essa informação" quando não tiver contexto nem resultado de ferramenta sobre o tema.

Contexto (políticas):
{contexto}

Pergunta:
{pergunta}
"""

    mensagens = [HumanMessage(content=prompt)]

    while True:
        resposta = llm_com_tools.invoke(mensagens)
        if not getattr(resposta, "tool_calls", None):
            saida = validate_output(resposta.content)
            print(f"\nAssistente: {saida}")
            break
        mensagens.append(resposta)
        for tc in resposta.tool_calls:
            fn = tools_por_nome.get(tc["name"])
            resultado = fn.invoke(tc["args"]) if fn else "Ferramenta não encontrada."
            mensagens.append(
                ToolMessage(content=str(resultado), tool_call_id=tc["id"])
            )
