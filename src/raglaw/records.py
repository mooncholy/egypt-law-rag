"""Read and write pipeline records under a schema header.

Every file names the model its records belong to and that model's schema
fingerprint (``Record.schema_fingerprint``), once, in a header:

- ``.json``: ``{"model": ..., "schema": ..., "records": [...]}``
- ``.jsonl``: the header object on the first line, then one record per line

Reading checks the header against the model before any record is parsed, so a
file written by older code fails loudly instead of loading into a newer schema.
Writing is deterministic (field order from the model, no ASCII escaping, a
trailing newline), so the same records always give the same bytes.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from raglaw.schema import Record


class SchemaMismatchError(ValueError):
    """A file's header names another model or schema than expected."""


def _header(model: type[Record]) -> dict[str, str]:
    return {"model": model.MODEL_NAME, "schema": model.schema_fingerprint()}


def _check_header(path: Path, header: object, model: type[Record]) -> None:
    expected = _header(model)
    found = {k: header.get(k) for k in expected} if isinstance(header, dict) else header
    if found != expected:
        raise SchemaMismatchError(
            f"{path}: header {found!r} does not match {model.schema_id()}; "
            "rebuild it with `dvc repro`"
        )


def write_records[R: Record](path: Path, model: type[R], records: Iterable[R]) -> int:
    """
    Write records under a header naming ``model`` and its schema fingerprint.

    The format follows the suffix: ``.jsonl`` writes one record per line,
    anything else one JSON document.

    returns:
    - count (int): how many records were written
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.model_dump(mode="json") for r in records]
    if path.suffix == ".jsonl":
        lines = [_header(model), *rows]
        text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    else:
        document = {**_header(model), "records": rows}
        text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(rows)


def read_records[R: Record](path: Path, model: type[R]) -> list[R]:
    """
    Read records written by ``write_records`` and validate each against ``model``.

    returns:
    - records (list[Record]): the file's records, in file order

    exceptions:
    - SchemaMismatchError: the header names another model or schema
    - pydantic.ValidationError: a record doesn't fit the model
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        header, *lines = text.splitlines()
        _check_header(path, json.loads(header), model)
        return [model.model_validate_json(line) for line in lines if line]
    document = json.loads(text)
    _check_header(path, document, model)
    return [model.model_validate(r) for r in document["records"]]
