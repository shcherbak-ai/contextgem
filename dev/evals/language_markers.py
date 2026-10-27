# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Tells the variant of a text in Chinese or Portuguese from its characters and words,
without a model: Simplified or Traditional Chinese, Japanese among texts that are
to be Chinese (by its kana), and Brazilian or European Portuguese.

The judge (``dev.evals.judge``) tells languages apart, but not these variants: in
the checks of 2026-10-27 it judged Traditional Chinese to be Simplified Chinese and
the reverse, European Portuguese to be Brazilian Portuguese and the reverse, and
a sentence in Japanese to be Traditional Chinese. The markers of a text are recorded
next to the verdict of the judge on its language.

The markers are evidence, not a verdict: a text without any of them, e.g. a short
value such as "Lei alemã", is written alike in both variants.
"""

from __future__ import annotations

import re
from typing import Literal

import langcodes


# What the markers of a text say about its variant: those of the required variant
# only, those of the other variant only, both, or none
VariantMarkers = Literal["required", "other", "mixed", "none"]

# The version of the markers, which the fingerprint of the judgments of the language
# includes (see `dev.evals.judge`): raise it when they change
MARKERS_VERSION = 1

# Pairs of a Simplified and a Traditional character, each of which is written only
# in its own script (characters that both scripts have, e.g. 于 and 后, are left out),
# among the common characters of contracts
_CHINESE_PAIRS = (
    "这這们們说說国國时時为為会會发發与與对對关關条條约約务務权權责責议議应應当當产產"
    "业業经經过過进進还還动動现現开開问問实實种種点點无無头頭体體两兩长長门門间間东東"
    "车車书書场場机機学學认認价價货貨运運输輸费費险險损損赔賠偿償签簽订訂数數购購买買"
    "卖賣销銷终終适適规規则則设設备備维維质質额額币幣汇匯账賬单單证證纳納该該将將处處"
    "让讓给給双雙优優负負际際协協违違赁賃讼訟诉訴决決担擔补補华華区區级級总總项項历歷"
)
_SIMPLIFIED = frozenset(_CHINESE_PAIRS[0::2])
_TRADITIONAL = frozenset(_CHINESE_PAIRS[1::2])

# Hiragana and katakana, which Japanese has and Chinese does not
_KANA = re.compile(r"[\u3040-\u30ff]")

# Words written differently in Brazilian and in European Portuguese, among
# the words of contracts: spellings (ônus and ónus, registro and registo,
# contato and contacto, fato and facto) and words (equipe and equipa, aditivo
# and adenda)
_PORTUGUESE_MARKERS = {
    "BR": re.compile(
        r"\b(ônus|econômic\w*|registros?|contatos?|fatos?|equipes?|arquivos?|"
        r"usuári\w*|prêmios?|gêneros?|anônim\w*|recepção|aditivos?|aluguel|"
        r"cadastr\w*|planejament\w*|gerenciament\w*|celular\w*)\b",
        re.IGNORECASE,
    ),
    "PT": re.compile(
        r"\b(ónus|económic\w*|registos?|contactos?|factos?|equipas?|ficheiros?|"
        r"utilizador\w*|prémios?|géneros?|anónim\w*|receção|adendas?|aluguer|"
        r"planeament\w*|telemóve\w*)\b",
        re.IGNORECASE,
    ),
}


def get_variant(tag: str) -> str | None:
    """
    Returns the variant of a language tag whose markers this module tells:
    "Hans" or "Hant" for Chinese, "BR" or "PT" for Portuguese.

    :param tag: The language tag, in its standard form.
    :type tag: str
    :return: The variant, or None for a tag of another language, or of Chinese
        or Portuguese without a script or region that tells the variant.
    :rtype: str | None
    """
    language = langcodes.Language.get(tag)
    if language.language == "zh":
        script = language.maximize().script if language.territory else language.script
        return script if script in ("Hans", "Hant") else None
    if language.language == "pt" and language.territory in ("BR", "PT"):
        return language.territory
    return None


def get_variant_markers(text: str, required_variant: str) -> VariantMarkers:
    """
    Tells which variant the markers of a text are of, against the variant
    required.

    :param text: The text.
    :type text: str
    :param required_variant: The variant required (see :func:`get_variant`).
    :type required_variant: str
    :return: "required" if the text has markers of the required variant only,
        "other" if it has markers of the other variant only (for Chinese, also
        if it has kana, i.e. is Japanese), "mixed" if it has both, "none" if
        it has none.
    :rtype: Literal["required", "other", "mixed", "none"]
    """
    if required_variant in ("Hans", "Hant"):
        if _KANA.search(text):
            return "other"
        found = {
            "Hans": any(char in _SIMPLIFIED for char in text),
            "Hant": any(char in _TRADITIONAL for char in text),
        }
    else:
        found = {
            variant: pattern.search(text) is not None
            for variant, pattern in _PORTUGUESE_MARKERS.items()
        }
    other_variant = next(variant for variant in found if variant != required_variant)
    if found[required_variant] and found[other_variant]:
        return "mixed"
    if found[required_variant]:
        return "required"
    if found[other_variant]:
        return "other"
    return "none"
