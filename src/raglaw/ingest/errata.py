"""Owner-approved fixes for one-off source errors (R7).

An erratum fixes one known error at one place, where a rule would be unsafe.
It names the row (page and row index), the side, and the **whole lines** to
replace: usually one, or several consecutive lines joined by ``\n`` when the
PDF split a word across a line break; ``""`` names an empty cell, so an entry
can fill a side the PDF left blank. Matching whole lines keeps an entry from
re-applying to its own fix (``rticle 452`` is a substring of ``Article 452``).
An entry whose lines aren't there fails the stage: a changed source is
stopped, never patched in the wrong place.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from raglaw.schema import Row


class ErratumNotFoundError(RuntimeError):
    """An erratum's expected line isn't at its page, row and side."""


class Erratum(BaseModel):
    """One entry of ``data/errata.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: int = Field(ge=1, description="Source page number, 1-based.")
    row: int = Field(ge=0, description="0-based row index in the page's table.")
    side: Literal["en", "ar"]
    expect: str = Field(
        description="A whole line of that cell, consecutive whole lines joined "
        'by `\\n`, or `""` for an empty cell.',
    )
    replace: str = Field(description="The line or lines that take their place.")
    reason: str = Field(min_length=1, description="Why, naming its fact (P).")


def load_errata(path: Path) -> list[Erratum]:
    """
    Read and validate the errata file.

    returns:
    - errata (list[Erratum]): every entry, in file order; an empty file has none
    """
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Erratum.model_validate(e) for e in entries]


def apply_erratum(row: Row, erratum: Erratum) -> Row:
    """
    Replace the one run of whole lines in ``row`` that equals ``erratum.expect``.

    returns:
    - row (Row): a copy with those lines replaced

    exceptions:
    - ErratumNotFoundError: the lines occur zero times, or more than once
    """
    field = f"{erratum.side}_text"
    lines = getattr(row, field).split("\n")
    expect = erratum.expect.split("\n")
    n = len(expect)
    hits = [i for i in range(len(lines) - n + 1) if lines[i : i + n] == expect]
    if len(hits) != 1:
        raise ErratumNotFoundError(
            f"page {erratum.page}, row {erratum.row}, {erratum.side}: expected "
            f"exactly one run of whole lines {erratum.expect!r}, found {len(hits)}"
        )
    i = hits[0]
    lines[i : i + n] = erratum.replace.split("\n")
    return row.model_copy(update={field: "\n".join(lines)})


def apply_errata(rows: Iterable[Row], errata: list[Erratum]) -> list[Row]:
    """
    Apply every erratum to its row.

    returns:
    - rows (list[Row]): all rows, in order, with each erratum applied once

    exceptions:
    - ErratumNotFoundError: an erratum's row or line isn't there
    """
    rows = list(rows)
    position = {(r.page, r.row_index): i for i, r in enumerate(rows)}
    for erratum in errata:
        key = (erratum.page, erratum.row)
        if key not in position:
            raise ErratumNotFoundError(
                f"page {erratum.page}, row {erratum.row}: no such row"
            )
        rows[position[key]] = apply_erratum(rows[position[key]], erratum)
    return rows
