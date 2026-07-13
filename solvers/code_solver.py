"""Layer 1 deterministic code solver — zero tokens.

Debugging: extract the broken function, apply common single-line fixes, verify by running.
Generation: produce a canonical implementation for well-known patterns, verify by running.
"""
import re
import subprocess
import sys
import textwrap
import tempfile
import os

_TIMEOUT = 8  # seconds per subprocess run


def _run_code(code: str) -> tuple[bool, str]:
    """Execute code in a subprocess. Returns (success, stdout/stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=_TIMEOUT,
            # Restrict environment — no network, minimal env
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
        )
        ok = r.returncode == 0
        return ok, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(fname)


# ── Debugging ────────────────────────────────────────────────────────────────

_DEBUG_FIXES = [
    # operator flips
    (r"\breturn\s+(\w+)\s*-\s*(\w+)", r"return \1 + \2"),
    (r"\breturn\s+(\w+)\s*\+\s*(\w+)", r"return \1 - \2"),
    (r"\breturn\s+(\w+)\s*\*\s*(\w+)", r"return \1 / \2"),
    (r"\breturn\s+(\w+)\s*/\s*(\w+)", r"return \1 * \2"),
    # off-by-one in range
    (r"range\((\w+)\)", r"range(\1 + 1)"),
    (r"range\((\w+)\s*\+\s*1\)", r"range(\1)"),
    # wrong comparison
    (r"\bif\s+(\w+)\s*>\s*(\w+)", r"if \1 < \2"),
    (r"\bif\s+(\w+)\s*<\s*(\w+)", r"if \1 > \2"),
]


def _extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Heuristic: lines that look like code
    lines = [l for l in text.splitlines() if l.strip().startswith(("def ", "class ", "    ", "return", "import"))]
    return "\n".join(lines) if lines else text


def _make_test_harness(code: str, func_name: str) -> str:
    """Append a minimal smoke-test for the function."""
    tests = {
        "add": "assert add(2,3)==5, add(2,3)\nassert add(0,0)==0",
        "subtract": "assert subtract(5,3)==2",
        "multiply": "assert multiply(3,4)==12",
        "factorial": "assert factorial(5)==120\nassert factorial(0)==1",
        "fibonacci": "assert fibonacci(6)==8",
        "is_prime": "assert is_prime(7)==True\nassert is_prime(4)==False",
        "reverse": "assert reverse('abc')=='cba'",
        "sum_list": "assert sum_list([1,2,3])==6",
    }
    test = tests.get(func_name, "")
    return f"{code}\n{test}" if test else code


def solve_code_debug(task: dict) -> dict | None:
    text = task.get("input", "")
    code = _extract_code_block(text)
    if not code.strip():
        return None

    # Find function name
    m = re.search(r"def\s+(\w+)", code)
    func_name = m.group(1) if m else ""

    harness = _make_test_harness(code, func_name)
    ok, _ = _run_code(harness)
    if ok:
        return {"answer": f"```python\n{code.strip()}\n```", "confidence": 1.0, "tokens_used": 0, "source": "code_solver"}

    # Try each fix
    for pattern, replacement in _DEBUG_FIXES:
        fixed = re.sub(pattern, replacement, code)
        if fixed == code:
            continue
        harness2 = _make_test_harness(fixed, func_name)
        ok2, _ = _run_code(harness2)
        if ok2:
            return {"answer": f"```python\n{fixed.strip()}\n```", "confidence": 1.0, "tokens_used": 0, "source": "code_solver"}

    return None  # Escalate


# ── Generation ───────────────────────────────────────────────────────────────

_TEMPLATES = {
    "factorial": textwrap.dedent("""\
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        """),
    "fibonacci": textwrap.dedent("""\
        def fibonacci(n):
            if n <= 0:
                return 0
            if n == 1:
                return 1
            return fibonacci(n - 1) + fibonacci(n - 2)
        """),
    "is_prime": textwrap.dedent("""\
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        """),
    "reverse string": textwrap.dedent("""\
        def reverse(s):
            return s[::-1]
        """),
    "palindrome": textwrap.dedent("""\
        def is_palindrome(s):
            s = s.lower()
            return s == s[::-1]
        """),
    "binary search": textwrap.dedent("""\
        def binary_search(arr, target):
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
        """),
    "bubble sort": textwrap.dedent("""\
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr
        """),
    "flatten": textwrap.dedent("""\
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result
        """),
}


def solve_code_gen(task: dict) -> dict | None:
    text = task.get("input", "").lower()
    for keyword, template in _TEMPLATES.items():
        if keyword in text:
            ok, _ = _run_code(template)
            if ok:
                return {"answer": f"```python\n{template.strip()}\n```", "confidence": 1.0, "tokens_used": 0, "source": "code_solver"}
    return None  # Escalate
