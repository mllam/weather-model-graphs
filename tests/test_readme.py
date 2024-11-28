import re
from pathlib import Path

import pytest


def _parse_readme_examples():
    lines = open(Path(__file__).parent.parent / "README.md").read().splitlines()

    # use regex to find all python code blocks
    code_blocks = re.findall(r"```python(.*?)```", "\n".join(lines), re.DOTALL)

    return code_blocks


@pytest.mark.parametrize("example", _parse_readme_examples())
def test_readme_example(example):
    exec(example)
