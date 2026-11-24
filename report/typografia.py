"""
Poprawki typograficzne tekstu LaTeX zgodne z zasadami składu polskiego.

- Jednoliterowe spójniki i przyimki (a, i, o, u, w, z) wiążemy z następnym
  słowem twardą spacją (~), żeby nie zostawały na końcu wiersza.
- Odwołania piszemy wielką literą i z twardą spacją: Tabela~\\ref{...},
  Rysunek~\\ref{...}, Sekcja~\\ref{...}, Dodatek~\\ref{...}.
- Twarda spacja przed \\cite i między nazwą wariantu a parametrem (S1~$r=1$).

Tryb matematyczny ($...$), komentarze i środowisko verbatim są pomijane.
Użycie: python report/typografia.py plik.tex [...]  (poprawia pliki w miejscu)
"""

import re
import sys

_REF_WORD = re.compile(r"\b(tabel|rysun|sekcj|dodat)(\w*)[ \t]*~?[ \t]*(?=\\ref)",
                       re.IGNORECASE)
_SINGLE = re.compile(r"(?<![\w\\~])([aiouwzAIOUWZ])[ \t]+(?=[\w$\\{(]|,,)")
# Przed wzorem w trybie matematycznym fragment tekstu kończy się spacją.
_SINGLE_END = re.compile(r"(?<![\w\\~])([aiouwzAIOUWZ])[ \t]+\Z")
_VARIANT_END = re.compile(r"\b(S[1-4])[ \t]+\Z")
_CITE = re.compile(r"(?<=\S)\\?[ \t]+(?=\\cite)")  # także "in.\ \cite"
_VARIANT = re.compile(r"\b(S[1-4])[ \t]+(?=\$)")


def _fix_text(chunk, before_math=False):
    chunk = _REF_WORD.sub(lambda m: m.group(1).capitalize() + m.group(2) + "~",
                          chunk)
    chunk = _SINGLE.sub(r"\1~", chunk)
    chunk = _CITE.sub("~", chunk)
    chunk = _VARIANT.sub(r"\1~", chunk)
    if before_math:
        chunk = _SINGLE_END.sub(r"\1~", chunk)
        chunk = _VARIANT_END.sub(r"\1~", chunk)
    return chunk


def _fix_line(line):
    code, sep, comment = _split_comment(line)
    parts = re.split(r"(\$[^$]*\$)", code)
    parts = [p if p.startswith("$") else _fix_text(p, i + 1 < len(parts))
             for i, p in enumerate(parts)]
    return "".join(parts) + sep + comment


def _split_comment(line):
    m = re.search(r"(?<!\\)%", line)
    if not m:
        return line, "", ""
    return line[:m.start()], "%", line[m.end():]


def popraw(text):
    out, verbatim = [], False
    for line in text.split("\n"):
        if "\\begin{verbatim}" in line:
            verbatim = True
        if verbatim:
            out.append(line)
        else:
            out.append(_fix_line(line))
        if "\\end{verbatim}" in line:
            verbatim = False
    return "\n".join(out)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        fixed = popraw(text)
        if fixed != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"poprawiono: {path}")
