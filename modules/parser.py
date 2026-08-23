from dataclasses import dataclass
import re


CLAUSE_PATTERN = re.compile(
    r"^\*\*(\d+(?:\.\d+){1,2})\*\*\s*(.*)$"
)

CROSS_REFERENCE_PATTERN = re.compile(
    r"§\d+(?:\.\d+){1,2}"
)

PART_PATTERN = re.compile(
    r"^# Part\s+(\d+)\s+—\s+(.+)$"
)

SECTION_PATTERN = re.compile(
    r"^##\s+(\d+\.\d+)\s+(.+)$"
)


@dataclass
class Clause:
    id: str
    part: str | None
    section: str | None
    text: str
    cross_references: list[str]


def parse_manual(markdown: str) -> list[Clause]:
    """
    Parse a policy manual into clause-level chunks.

    A clause begins with a numbered provision such as:
        **4.3.2** A recipient must...

    Everything until the next numbered provision belongs to
    the current clause.
    """

    lines = markdown.splitlines()

    clauses: list[Clause] = []

    current_part: str | None = None
    current_section: str | None = None

    current_id: str | None = None
    current_text: list[str] = []

    def save_current_clause() -> None:
        nonlocal current_id, current_text

        if current_id is None:
            return

        text = "\n".join(current_text).strip()

        cross_references = sorted(
            set(CROSS_REFERENCE_PATTERN.findall(text))
        )

        clauses.append(
            Clause(
                id=f"§{current_id}",
                part=current_part,
                section=current_section,
                text=text,
                cross_references=cross_references,
            )
        )

        current_id = None
        current_text = []

    for line in lines:
        part_match = PART_PATTERN.match(line.strip())

        if part_match:
            save_current_clause()

            part_number = part_match.group(1)
            part_title = part_match.group(2)

            current_part = f"Part {part_number} — {part_title}"
            current_section = None
            continue

        section_match = SECTION_PATTERN.match(line.strip())

        if section_match:
            save_current_clause()

            current_section = (
                f"{section_match.group(1)} {section_match.group(2)}"
            )
            continue

        clause_match = CLAUSE_PATTERN.match(line.strip())

        if clause_match:
            save_current_clause()

            current_id = clause_match.group(1)
            first_line = clause_match.group(2)

            if first_line:
                current_text.append(first_line)

            continue

        if current_id is not None:
            current_text.append(line)

    save_current_clause()

    return clauses