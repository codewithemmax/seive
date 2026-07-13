"""Layer 1 deterministic math solver — zero tokens."""
import re
from sympy import sympify, solve, sqrt, symbols, SympifyError  # noqa: F401
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCAL = {"sqrt": sqrt}


def _try_evaluate(expr_str: str):
    """Return numeric result or None."""
    try:
        expr = parse_expr(expr_str, local_dict=_LOCAL, transformations=_TRANSFORMS)
        result = expr.evalf()
        # Return int if whole number, else float rounded to 6 sig figs
        if result == int(result):
            return int(result)
        return float(f"{float(result):.6g}")
    except Exception:
        return None


def _try_solve(text: str):
    """Detect 'solve for x: ...' pattern and return solutions."""
    m = re.search(r"solve\s+for\s+(\w+)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    var_name, eq_str = m.group(1).strip(), m.group(2).strip()
    x = symbols(var_name)
    # Normalise: treat '=' as lhs - rhs = 0
    if "=" in eq_str:
        lhs, rhs = eq_str.split("=", 1)
        try:
            expr = parse_expr(lhs, local_dict={var_name: x, "sqrt": sqrt}, transformations=_TRANSFORMS) \
                 - parse_expr(rhs, local_dict={var_name: x, "sqrt": sqrt}, transformations=_TRANSFORMS)
        except Exception:
            return None
    else:
        try:
            expr = parse_expr(eq_str, local_dict={var_name: x, "sqrt": sqrt}, transformations=_TRANSFORMS)
        except Exception:
            return None
    try:
        sols = solve(expr, x)
        if not sols:
            return None
        return ", ".join(str(s.evalf() if s.is_real else s) for s in sols)
    except Exception:
        return None


def solve_math(task: dict) -> dict | None:
    text = task.get("input", "")

    # Try equation solving first
    sol = _try_solve(text)
    if sol is not None:
        return {"answer": sol, "confidence": 1.0, "tokens_used": 0, "source": "math_solver"}

    # Strip question words and try direct evaluation
    expr_str = re.sub(r"[Ww]hat\s+is\s+|[Cc]alculate\s+|[Ee]valuate\s+|\?", "", text).strip()
    result = _try_evaluate(expr_str)
    if result is not None:
        return {"answer": str(result), "confidence": 1.0, "tokens_used": 0, "source": "math_solver"}

    return None
