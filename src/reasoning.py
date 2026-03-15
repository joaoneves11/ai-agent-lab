
REASONING_INSTRUCTION = (
    "Antes de responder ou de usar uma ferramenta, raciocine em uma frase sobre o que fazer."
)

USE_REASONING_TAGS = False

REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"


def inject_reasoning_instruction(prompt: str, use_tags: bool = False) -> str:
    """Adiciona a instrução de reasoning no início do prompt."""
    instruction = REASONING_INSTRUCTION
    if use_tags and USE_REASONING_TAGS:
        instruction = (
            f"Se for explicar seu raciocínio, coloque entre {REASONING_OPEN} e {REASONING_CLOSE}. "
            "Depois responda em texto normal para o usuário."
        )
    return f"{instruction}\n\n{prompt}"


def extract_reasoning_and_response(text: str) -> tuple[str, str]:
    """
    Se o texto tiver <reasoning>...</reasoning>, retorna (reasoning, resposta_limpa).
    Senão retorna ("", text).
    """
    if not text or REASONING_OPEN not in text or REASONING_CLOSE not in text:
        return "", text or ""
    start = text.find(REASONING_OPEN) + len(REASONING_OPEN)
    end = text.find(REASONING_CLOSE)
    reasoning = text[start:end].strip() if end != -1 else ""
    response = (
        text[: text.find(REASONING_OPEN)].strip() + " " + text[end + len(REASONING_CLOSE) :].strip()
    ).strip() if end != -1 else text
    return reasoning, response or text
