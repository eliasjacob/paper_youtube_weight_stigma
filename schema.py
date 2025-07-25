from typing import Literal
from pydantic import BaseModel, Field

class RespostaAnaliseSentimento(BaseModel):
    """A resposta de uma função que realiza análise de sentimento em texto e detecção do idioma do texto."""

    # O rótulo de sentimento atribuído ao texto
    sentimento: Literal["positivo", "negativo", "neutro"] = Field(
        default_factory=str,
        description="O rótulo de sentimento atribuído ao texto. Você só pode ter 'positivo', 'negativo' ou 'neutro' como valores.",
    )

    gordofobia_implicita: bool = Field(
        default_factory=bool,
        description="Se o texto contém discriminação por peso (gordofobia) de forma implícita e/ou indireta. Se não houver gordofobia, este campo deve ser False.",
    )

    gordofobia_explicita: bool = Field(
        default_factory=bool,
        description="Se o texto contém discriminação por peso (gordofobia) de forma explícita e/ou direta. Se não houver gordofobia, este campo deve ser False.",
    )

    # O idioma detectado no texto
    idioma: str = Field(
        default_factory=str,
        description="O idioma detectado no texto, representado por um código de idioma de duas letras.",
    )

    obesidade: bool = Field(
        default_factory=bool,
        description="Se o texto toca no assunto de obesidade. Se não houver menção à obesidade, este campo deve ser False.",
    )
    