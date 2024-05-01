from pathlib import Path
import re

def main():
    # parse ../README.md, find first python code snippet and then execute it
    fp_readme = Path(__file__).parent.parent / "README.md"
    
    # read README.md
    with open(fp_readme, "r") as f:
        readme = f.read()

    # find first python code snippet
    code = re.search(r"```python\n(.*?)\n```", readme, re.DOTALL)
    code = code.group(1)

    # execute code
    exec(code)
    

if __name__ == "__main__":
    main()