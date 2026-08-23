from pathlib import Path

from modules.parser import parse_manual


def test_parser_basic_structure():
    manual_path = Path("data/policy-manual.md")
    markdown = manual_path.read_text(encoding="utf-8")

    clauses = parse_manual(markdown)

    assert len(clauses) > 0


def test_clause_2_1_2_keeps_lettered_items():
    manual_path = Path("data/policy-manual.md")
    markdown = manual_path.read_text(encoding="utf-8")

    clauses = parse_manual(markdown)

    clause = next(c for c in clauses if c.id == "§2.1.2")

    assert "(a)" in clause.text
    assert "(b)" in clause.text
    assert "(c)" in clause.text
    assert "(d)" in clause.text
    assert "(e)" in clause.text
    assert "(f)" in clause.text


def test_cross_references_are_extracted():
    manual_path = Path("data/policy-manual.md")
    markdown = manual_path.read_text(encoding="utf-8")

    clauses = parse_manual(markdown)

    clause = next(c for c in clauses if c.id == "§9.1.4")

    assert "§4.3" in clause.cross_references