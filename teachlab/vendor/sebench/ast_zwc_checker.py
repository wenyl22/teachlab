import ast
import sys
from typing import Any, Dict, List, Set, Tuple, Optional


class ASTSourceValidator:
    def __init__(self, lib_name: str = "zwc"):
        self.lib_name = lib_name
        self.imported_names: Set[str] = set()

    def check_source(self, source_code: str, entry_point: str, *args, **kwargs) -> dict:
        report = {
            "passed": False,
            "library_imported": False,
            "library_called": False,
            "return_value_tainted": False,
            "meaningful_usage": False,
            "error": None,
            "call_details": [],
            "return_statements_analyzed": [],
        }

        try:
            tree = ast.parse(source_code)

            self._analyze_imports(tree, report)

            if not report["library_imported"]:
                report["error"] = f"No {self.lib_name} module"
                return report

            function_node = self._find_function(tree, entry_point)
            if not function_node:
                report["error"] = f"Unfound function '{entry_point}'"
                return report

            self._comprehensive_analysis(function_node, report)

            if (
                report["library_imported"]
                and report["library_called"]
                and report["return_value_tainted"]
                and report["meaningful_usage"]
            ):
                report["passed"] = True
            else:
                reasons = []
                if not report["library_called"]:
                    reasons.append("Do not contain library calls")
                if not report["return_value_tainted"]:
                    reasons.append("Return value independent of target liarary")
                if not report["meaningful_usage"]:
                    reasons.append("Meaningless calls")
                report["error"] = f"Failed: {', '.join(reasons)}"

        except SyntaxError as e:
            report["error"] = f"Code syntax error: {e}"
        except Exception as e:
            report["error"] = f"AST Analysis error: {e}"

        return report

    def _analyze_imports(self, tree: ast.AST, report: dict):
        """Analyse sentence"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == self.lib_name:
                        report["library_imported"] = True
                        imported_name = alias.asname if alias.asname else alias.name
                        self.imported_names.add(imported_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == self.lib_name:
                    report["library_imported"] = True
                    for alias in node.names:
                        imported_name = alias.asname if alias.asname else alias.name
                        self.imported_names.add(imported_name)

    def _find_function(
        self, tree: ast.AST, function_name: str
    ) -> Optional[ast.FunctionDef]:
        """Find target definitions in AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None

    def _comprehensive_analysis(self, function_node: ast.FunctionDef, report: dict):
        """Spetial cases"""
        library_calls = []
        return_statements = []

        for node in ast.walk(function_node):
            if self._is_library_call(node):
                call_info = self._extract_call_info(node)
                library_calls.append(call_info)
                report["call_details"].append(call_info)

            if isinstance(node, ast.Return):
                return_statements.append(node)

        report["library_called"] = len(library_calls) > 0

        if not report["library_called"]:
            return

        tainted_returns = []
        for ret_node in return_statements:
            analysis = self._analyze_return_statement(ret_node, library_calls)
            report["return_statements_analyzed"].append(analysis)
            if analysis["is_tainted"]:
                tainted_returns.append(analysis)

        report["return_value_tainted"] = len(tainted_returns) > 0

        report["meaningful_usage"] = self._has_meaningful_usage(library_calls)

    def _analyze_return_statement(
        self, return_node: ast.Return, library_calls: List[dict]
    ) -> Dict[str, Any]:
        """Analyse single sentence"""
        analysis = {
            "line": return_node.lineno,
            "expression": (
                ast.unparse(return_node.value) if return_node.value else "None"
            ),
            "is_tainted": False,
            "taint_reason": None,
            "direct_library_call": False,
            "variable_dependencies": [],
        }

        if return_node.value is None:
            return analysis

        if self._is_library_call(return_node.value):
            analysis["is_tainted"] = True
            analysis["direct_library_call"] = True
            analysis["taint_reason"] = "Returned liarary call"
            return analysis

        if isinstance(return_node.value, ast.Name):
            var_name = return_node.value.id
            analysis["is_tainted"] = True 
            analysis["taint_reason"] = f"Varient '{var_name}' contained meaningful library calls"
            analysis["variable_dependencies"].append(var_name)
            return analysis

        if self._contains_library_call(return_node.value):
            analysis["is_tainted"] = True
            analysis["taint_reason"] = "Contained meaningful library calls"
            return analysis

        if isinstance(return_node.value, (ast.List, ast.Tuple)):
            elements_tainted = False
            for element in return_node.value.elts:
                if self._contains_library_call(element):
                    elements_tainted = True
                    break
            if elements_tainted:
                analysis["is_tainted"] = True
                analysis["taint_reason"] = "Contained meaningful library calls"
                return analysis

        analysis["is_tainted"] = True
        analysis["taint_reason"] = "Inderect Dependence"

        return analysis

    def _is_library_call(self, node: ast.AST) -> bool:
        """Judge if the called library is the target libaray"""
        if not isinstance(node, ast.Call):
            return False

        func = node.func

        # tackle zwc.function()
        if isinstance(func, ast.Attribute):
            current = func.value
            while isinstance(current, ast.Attribute):
                current = current.value

            if isinstance(current, ast.Name):
                return current.id in self.imported_names

        elif isinstance(func, ast.Name):
            return func.id in self.imported_names

        return False

    def _contains_library_call(self, node: ast.AST) -> bool:
        """Check if the expression is meaningful"""
        if self._is_library_call(node):
            return True

        for child in ast.iter_child_nodes(node):
            if self._contains_library_call(child):
                return True

        return False

    def _extract_call_info(self, call_node: ast.Call) -> Dict[str, Any]:
        """Get calls infomation"""
        call_info = {
            "line": call_node.lineno,
            "function": self._get_function_name(call_node.func),
            "args": len(call_node.args),
            "keywords": len(call_node.keywords),
        }

        if isinstance(call_node.func, ast.Attribute):
            call_info["method"] = call_node.func.attr

        return call_info

    def _get_function_name(self, func_node: ast.AST) -> str:
        """Get str of function name"""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return f"{self._get_function_name(func_node.value)}.{func_node.attr}"
        return "unknown"

    def _has_meaningful_usage(self, library_calls: List[dict]) -> bool:
        """Judge if the library calls is meaningful"""
        if not library_calls:
            return False

        meaningless_patterns = [
            lambda call: "yitaf" in call.get("function", "").lower()
            and len(library_calls) == 1,
        ]

        # calls is judged meaningless only if calls are sinple convertion
        all_meaningless = all(
            any(pattern(call) for pattern in meaningless_patterns)
            for call in library_calls
        )

        return not all_meaningless


if __name__ == "__main__":
    validator = ASTSourceValidator("zwc")

    print("Example: should fail")
    code_no_lib = """
import math
import zwc
def solve(x):
    return math.cos(x + 1)
"""
    result = validator.check_source(code_no_lib, "solve")
    print(f"Passed: {result['passed']}")
    print(f"Error: {result['error']}\n")
