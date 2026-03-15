
from __future__ import annotations

import re

ALLOWED_TOPICS = [
    "férias",
    "reembolso",
    "viagens",
    "despesas",
    "gastos",
    "política",
    "financeiro",
    "rh",
    "total",
    "ano",
]


BLOCKED_TERMS = [
    "salário de todos",
    "delete",
    "drop table",
    "hackear",
    "senha",
]


def validate_input(user_input: str) -> tuple[bool, str | None]:
  
    if not user_input or not user_input.strip():
        return False, "Por favor, digite uma pergunta."

    lower = user_input.lower().strip()

    for term in BLOCKED_TERMS:
        if term in lower:
            return False, "Pergunta bloqueada por política de segurança."

    if any(topic in lower for topic in ALLOWED_TOPICS):
        return True, None
    return False, "Posso ajudar apenas com políticas internas e consultas autorizadas."


def validate_output(text: str) -> str:
   
    if not text:
        return text
    result = text
    for term in BLOCKED_TERMS:
        result = re.sub(re.escape(term), "[removido]", result, flags=re.IGNORECASE)
    return result
