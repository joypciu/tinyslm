"""Creative coding scaffolds — structure without free neural invent.

When no FAQ/code card matches, compose a small CLI / function+tests skeleton
from constraints in the user ask. Always meant to be verified via Spec-Assert
+ syntax before shipping.
"""

from __future__ import annotations

import re
from typing import Optional

from tiny_slm.code_verify import parse_io_examples


def compose_scaffold(user: str) -> Optional[str]:
    """Return a small verified-oriented scaffold, or None if too vague."""
    u = (user or "").strip()
    ulow = u.lower()
    if not u:
        return None
    # Need a coding-shaped ask
    if not any(
        w in ulow
        for w in (
            "python",
            "function",
            "implement",
            "write a",
            "script",
            "cli",
            "command",
            "def ",
        )
    ):
        return None

    examples = parse_io_examples(u)
    # Function + tests from examples
    if examples:
        fn = examples[0][0]
        # Guess arity from first example args
        n_args = len([a for a in examples[0][1].split(",") if a.strip()]) if examples[0][1] else 0
        params = ", ".join(f"a{i}" for i in range(max(1, n_args)))
        body = (
            f"def {fn}({params}):\n"
            f"    # TODO: implement to satisfy examples\n"
            f"    raise NotImplementedError\n"
        )
        # Prefer a trivial implement when examples look like add
        if fn == "add" and n_args == 2:
            body = f"def {fn}(a0, a1):\n    return a0 + a1\n"
        notes = "\n".join(
            f"# example: {fn}({args}) -> {exp}" for fn, args, exp in examples[:3]
        )
        return body + "\n" + notes

    # CLI scaffold
    if any(w in ulow for w in ("cli", "command line", "argparse", "subcommand")):
        return (
            "import argparse\n\n"
            "def main(argv=None):\n"
            "    p = argparse.ArgumentParser(description='tiny CLI')\n"
            "    p.add_argument('path', nargs='?', default='.')\n"
            "    p.add_argument('--dry-run', action='store_true')\n"
            "    args = p.parse_args(argv)\n"
            "    print('path', args.path, 'dry_run', args.dry_run)\n"
            "    return 0\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )

    # Generic pure function scaffold when "write a function" but no card
    m = re.search(
        r"(?:function|def)\s+(?:named\s+)?([a-zA-Z_]\w*)|"
        r"write a (?:python )?function (?:that |to |which )?([a-zA-Z_]\w*)",
        ulow,
    )
    if m:
        fn = m.group(1) or m.group(2) or "solve"
        return (
            f"def {fn}(*args, **kwargs):\n"
            f"    # Scaffold: replace with a verified implementation\n"
            f"    raise NotImplementedError\n"
        )
    return None
