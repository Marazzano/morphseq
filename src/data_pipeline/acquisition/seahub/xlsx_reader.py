"""Small dependency-free XLSX reader for the SeaHub metadata workbook."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _column_number(cell_reference: str) -> int:
    match = re.match(r"[A-Z]+", cell_reference.upper())
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    number = 0
    for letter in match.group(0):
        number = number * 26 + ord(letter) - ord("A") + 1
    return number - 1


def _unique_headers(values: Sequence[Any]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values):
        base = str(value).strip() if value not in (None, "") else f"column_{index}"
        counts[base] = counts.get(base, 0) + 1
        headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return headers


def read_xlsx_sheet(path: str | Path, *, sheet_name: str) -> pd.DataFrame:
    """Read the value grid of one worksheet using the Python standard library.

    SeaHub's runtime environment intentionally does not require ``openpyxl``.
    This reader supports the cell encodings present in the authoritative
    collection workbook: shared/inline strings, booleans, and cached numerics.
    """
    with ZipFile(Path(path)) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{_MAIN_NS}t"))
                for item in shared_root.findall(f"{_MAIN_NS}si")
            ]

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in rels_root.findall(f"{_PKG_REL_NS}Relationship")
        }
        sheet_target: str | None = None
        for sheet in workbook_root.findall(f".//{_MAIN_NS}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                sheet_target = targets[sheet.attrib[f"{_REL_NS}id"]]
                break
        if sheet_target is None:
            available = [
                sheet.attrib.get("name")
                for sheet in workbook_root.findall(f".//{_MAIN_NS}sheet")
            ]
            raise KeyError(
                f"Worksheet {sheet_name!r} not found; available={available}"
            )

        xml_path = (
            sheet_target.lstrip("/")
            if sheet_target.startswith("/")
            else f"xl/{sheet_target}"
        )
        sheet_root = ET.fromstring(archive.read(xml_path))
        sparse_rows: list[dict[int, Any]] = []
        max_column = -1
        for row in sheet_root.iter(f"{_MAIN_NS}row"):
            values: dict[int, Any] = {}
            for cell in row.findall(f"{_MAIN_NS}c"):
                column = _column_number(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{_MAIN_NS}v")
                value: Any = None if value_node is None else value_node.text
                if cell_type == "s" and value is not None:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{_MAIN_NS}t")
                    )
                elif cell_type == "b" and value is not None:
                    value = value == "1"
                elif cell_type in (None, "n") and value not in (None, ""):
                    try:
                        numeric = float(value)
                        value = int(numeric) if numeric.is_integer() else numeric
                    except ValueError:
                        pass
                values[column] = value
                max_column = max(max_column, column)
            if values and any(value not in (None, "") for value in values.values()):
                sparse_rows.append(values)

    if not sparse_rows:
        return pd.DataFrame()
    rows = [
        [sparse.get(column) for column in range(max_column + 1)]
        for sparse in sparse_rows
    ]
    return pd.DataFrame(rows[1:], columns=_unique_headers(rows[0]))


__all__ = ["read_xlsx_sheet"]
