from pathlib import Path

from langchain_core.tools import tool


@tool
def consultar_gastos(mes: str) -> str:
    """Consulta o total de gastos de um mês. Use quando a pessoa perguntar quanto gastamos, despesas do mês, etc. Exemplo de mês: 2026-01, 2026-02."""
    import json
    path = Path(__file__).resolve().parent.parent / "data" / "expenses.json"
    if not path.exists():
        return "Arquivo de despesas não encontrado."
    dados = json.loads(path.read_text())
    for item in dados:
        if item.get("month") == mes:
            return f"Gastos em {mes}: R$ {item['total']:,.2f}"
    return f"Não há dados para o mês {mes}. Meses disponíveis: {[d['month'] for d in dados]}"


@tool
def listar_meses_com_gastos() -> str:
    """Lista todos os meses que têm registro de gastos. Use quando a pessoa perguntar quais meses temos, ou períodos disponíveis."""
    import json
    path = Path(__file__).resolve().parent.parent / "data" / "expenses.json"
    if not path.exists():
        return "Arquivo de despesas não encontrado."
    dados = json.loads(path.read_text())
    meses = [d["month"] for d in dados]
    return "Meses com gastos: " + ", ".join(meses)
