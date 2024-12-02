import re
from pathlib import Path

import pytest


def _parse_readme_examples():
    lines = open(Path(__file__).parent.parent / "README.md").read().splitlines()

    # use regex to find all python code blocks
    code_blocks = re.findall(r"```python(.*?)```", "\n".join(lines), re.DOTALL)

    return code_blocks


@pytest.mark.parametrize("codeblock_example", _parse_readme_examples())
def test_readme_example(codeblock_example: str):
    """
    Check that execution of the python code block in the README does not raise an exception.
    """
    exec(codeblock_example)
