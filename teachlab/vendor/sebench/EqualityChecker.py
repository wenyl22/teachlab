import re
import ast
from typing import Any, List, Tuple, Union
import numpy as np
import math

DEBUG = False

def analyze_source(src: str, your_pkg="mynp"):
    tree = ast.parse(src)
    aliases = {} 
    used_numpy = False
    used_yourpkg = False

    class V(ast.NodeVisitor):
        def visit_Import(self, node):
            nonlocal used_numpy, used_yourpkg
            for a in node.names:
                if a.name == "numpy" or a.name.startswith("numpy."):
                    used_numpy = True
                if a.name == your_pkg or a.name.startswith(your_pkg + "."):
                    used_yourpkg = True
                if a.asname:
                    aliases[a.asname] = a.name
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            nonlocal used_numpy, used_yourpkg
            if node.module:
                if node.module == "numpy" or node.module.startswith("numpy."):
                    used_numpy = True
                if node.module == your_pkg or node.module.startswith(your_pkg + "."):
                    used_yourpkg = True
            self.generic_visit(node)

        def visit_Call(self, node):
            # catch __import__("numpy")
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and isinstance(node.args[0], ast.Str):
                    if node.args[0].s.startswith("numpy"):
                        used_numpy = True
            self.generic_visit(node)

    V().visit(tree)
    return used_numpy, used_yourpkg


class EqulityChecker:
    def __init__(self, tolerance: float = 1e-5):
        self._tolerance = tolerance

    def parse_stdout_output(self, output_str: str) -> Any:
        """
        Parse str to python object.
        Tackle form of 'print()' output.
        """
        if not output_str.strip():
            return None

        # clean str
        cleaned = output_str.strip()

        # take out array
        cleaned = self.process_empty_arrays(cleaned)

        # tackle Multiline array
        cleaned = self.process_multiline_arrays(cleaned)

        # tackle NumPy array
        cleaned = self.process_numpy_arrays(cleaned)

        # tackle ZWCArray
        cleaned = self.process_custom_arrays(cleaned)

        # clean charactors
        cleaned = self.extract_largest_brackets(cleaned)

        # clean ' '
        cleaned = self.process_spaces(cleaned)

        # convert 'inf' and 'nan'
        cleaned = cleaned.replace("inf", 'float("inf")').replace("nan", 'float("nan")')

        # try to convert to python object
        try:
            return eval(cleaned)
        except:
            # if failed, compare cleaned
            return cleaned

    def process_multiline_arrays(self, text: str) -> str:
        if "\n" in text:
            # [[1 2]\n [3 4]] -> [[1, 2], [3, 4]]
            lines = re.split(r"\n+", text)
            lines = [line.strip() for line in lines if line.strip()]
            lines = [line + "," if not line[-1] == "," else line for line in lines]
            merged = " ".join(lines)
            return merged

        return text

    def process_numpy_arrays(self, text: str) -> str:
        # array([1, 2, 3]) -> [1, 2, 3]
        text = re.sub(r"array", r" ", text)

        # [1 2 3] -> [1, 2, 3]
        def add_commas(match):
            content = match.group(1)
            elements = content.split()
            if elements:
                elements = [elem.strip() for elem in elements]
                new_content = ", ".join(elements)
                return f"[{new_content}]"
            else:
                return match.group(0)

        text = re.sub(r"\[([\d\s\.eE+\-infnanTrueFalsej\(\)]+)\]", add_commas, text)
        return text

    def process_custom_arrays(self, text: str) -> str:
        """tackle ZWCArray"""
        text = re.sub(r"ZWCArray", r" ", text)
        return text

    def process_spaces(self, text: str) -> str:
        text = re.sub(r"\s+", "", text)
        text = re.sub(r",", ", ", text)
        return text

    def process_empty_arrays(self, text: str) -> str:
        # array([], dtype=float64) -> []
        text = re.sub(r"array\(\s*\[\s*\],\s*dtype=[\w\']+\s*\)", "[]", text)

        # array([], shape=(0, 3), dtype=float64) -> []
        text = re.sub(
            r"array\(\s*\[\s*\],\s*shape=\([^)]*\),\s*dtype=[\w\']+\s*\)", "[]", text
        )

        # ZWCArray([]) -> []
        text = re.sub(r"ZWCArray\(\s*\[\s*\]\s*\)", "[]", text)
        text = re.sub(r"\w+Array\(\s*\[\s*\]\s*\)", "[]", text)

        return text

    def extract_largest_brackets(self, text: str) -> str:
        if ":" in text:
            colon_index = text.index(":")
            text = text[colon_index + 1 :].strip()
        return text

    def process_complex_structures(self, text: str) -> Any:
        # (array([1,2]), [3,4])
        if "(" in text and ")" in text:
            simplified = text
            simplified = re.sub(r"array\(([^)]+)\)", r"\1", simplified)
            simplified = re.sub(r"\w+Array\(([^)]+)\)", r"\1", simplified)
            simplified = simplified.replace("\n", " ")
            try:
                return eval(
                    simplified.replace("inf", 'float("inf")').replace(
                        "nan", 'float("nan")'
                    )
                )
            except (ValueError, SyntaxError):
                pass
        
        return text

    def normalize_data_structure(self, obj: Any) -> Any:
        if obj is None:
            return None

        if isinstance(obj, str):
            try:
                processed = self.process_multiline_arrays(obj)
                processed = self.process_numpy_arrays(processed)
                processed = self.process_custom_arrays(processed)
                reparsed = ast.literal_eval(processed)
                return self.normalize_data_structure(reparsed)
            except:
                return obj

        if isinstance(obj, list):
            return [self.normalize_data_structure(item) for item in obj]

        if isinstance(obj, tuple):
            return [self.normalize_data_structure(item) for item in obj]

        if hasattr(obj, "tolist"):
            return self.normalize_data_structure(obj.tolist())

        if hasattr(obj, "item"):
            return obj.item()

        return obj

    def _unwrap_single_element(self, obj):
        while isinstance(obj, (list, tuple, np.ndarray)) and len(obj) == 1:
            obj = obj[0]
        return obj

    def deep_compare(
        self, a: Any, b: Any, tolerance: float = 1e-10, path: str = ""
    ) -> bool:
        a = self._unwrap_single_element(a)
        b = self._unwrap_single_element(b)

        if a is None or b is None:
            return a == b

        a_type = type(a)
        b_type = type(b)

        if a_type != b_type:

            if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
                b, (int, float, np.integer, np.floating)
            ):
                pass

            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                pass
            else:
                return False

        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
            b, (int, float, np.integer, np.floating)
        ):
            if a == float("inf") and b == float("inf"):
                return True
            if a == float("-inf") and b == float("-inf"):
                return True
            if math.isnan(a) and math.isnan(b):
                return True
            return abs(float(a) - float(b)) <= tolerance

        if isinstance(a, str) and isinstance(b, str):
            try:
                a_parsed = self.parse_stdout_output(a)
                b_parsed = self.parse_stdout_output(b)
                if a_parsed != a or b_parsed != b:
                    return self.deep_compare(a_parsed, b_parsed, tolerance)
            except:
                pass
            return a == b

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False

            for i, (item_a, item_b) in enumerate(zip(a, b)):
                if not self.deep_compare(item_a, item_b, tolerance, f"{path}[{i}]"):
                    return False
            return True

        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            for key in a.keys():
                if not self.deep_compare(a[key], b[key], tolerance):
                    return False
            return True

        return a == b

    def compare_stdout_outputs(
        self, pred_stdout: str, target_stdout: str, tolerance: float = 1e-10
    ) -> bool:
        # parse output
        pred_parsed = self.parse_stdout_output(pred_stdout)
        target_parsed = self.parse_stdout_output(target_stdout)

        pred_normalized = self.normalize_data_structure(pred_parsed)
        target_normalized = self.normalize_data_structure(target_parsed)

        try:
            result = self.deep_compare(pred_normalized, target_normalized, tolerance)
            if not result and DEBUG:
                print(f"-----------------------------------------")
                print(f"Failed Comparation:")
                print(f"Pred stdout: {pred_stdout}")
                print(f"Predicted normalized: {pred_normalized}")
                print(f"-----")
                print(f"Target stdout: {target_stdout}")
                print(f"Target normalized: {target_normalized}")
            return result
        except Exception as e:
            print(e)
            print(
                f"pred:{pred_stdout}\ntarget:{target_stdout}\n Error while comparation, return str comparation."
            )
            pred_clean = re.sub(r"\s+", " ", pred_stdout.strip())
            target_clean = re.sub(r"\s+", " ", target_stdout.strip())
            return pred_clean == target_clean

    def judge(self, exe_result: list, ground_truth_list: list):
        """Compare if all the cases in list are the same"""
        try:
            for i in range(len(ground_truth_list)):
                pred = exe_result[i]
                target = ground_truth_list[i]
                result = pred == target or self.compare_stdout_outputs(
                    pred, target, self._tolerance
                )
                if result == False:
                    return False
            return True
        except Exception as e:
            print(e)
            breakpoint()
            return False

    # Test function
    def test_comparison(self):

        test_cases = [

            ("[1, 2, 3]", "[1, 2, 3]", True),
            ("[1, 2, 3]", "[1, 2, 4]", False),

            ("[[1 2]\n [3 4]]", "[[1, 2], [3, 4]]", True),
            ("[2 1]", "[2, 1]", True),

            ("(array([1, 2]), [3, 4])", "([1, 2], [3, 4])", True),
            ("([1, 2], array([3, 4]))", "([1, 2], [3, 4])", True),
            ("([1, 2], ZWCArray([3, 4]))", "([1, 2], [3, 4])", True),

            ("[(1, 2), (3, 4)]", "[[1, 2], [3, 4]]", True),
            ("([(1, 2)], [array([3, 4])])", "([[1, 2]], [[3, 4]])", True),

            ("array([1, 2, 3])", "[1, 2, 3]", True),
            ("ZWCArray([1, 2, 3])", "[1, 2, 3]", True),

            ("[[1 2 3]\n [4 5 6]]", "[[1, 2, 3], [4, 5, 6]]", True),
            ("[[1]\n [2]\n [3]]", "[[1], [2], [3]]", True),
            (
                "[[1.00000000e+00 0.00000000e+00 0.00000000e+00]\n [0.00000000e+00 6.32455532e-03 0.00000000e+00]\n [0.00000000e+00 0.00000000e+00 1.11111111e-04]]",
                "[[1.00000000e+00 0.00000000e+00 0.00000000e+00]\n [0.00000000e+00 6.32455532e-03 0.00000000e+00]\n [0.00000000e+00 0.00000000e+00 1.11111111e-04]]",
                True,
            ),

            ("[3.56 3.57]", "[3.56 3.57]", True),
            ("3.567", "3.568", True),

            (
                r"{'key1': array([1, 2, 3]), 'answer': ZWCArray([3, 2, 1])}",
                r"{'key1': ZWCArray([1, 2, 3]), 'answer': array([3, 2, 1])}",
                True,
            ),

            ("[inf, -inf, nan]", "[inf, -inf, nan]", True),
            ("array([inf, -inf, nan])", "[inf, -inf, nan]", True),
            ("[1.0, inf, nan]", "[1.0, inf, nan]", True),

            ("[True False True]", "[True, False, True]", True),

            ("[1e-09, 1e+09]", "[1e-09, 1e+09]", True),
            ("array([1e-09, 1e+09])", "[1e-09, 1e+09]", True),
            (
                "[1.00000000e+00 0.00000000e+00 0.00000000e+00]",
                "[1.00000000e+00 0.00000000e+00 0.00000000e+00]",
                True,
            ),

            ("array([1. , 2.5, 4. ])", "[1.  2.5 4. ]", True),
            ("array([], dtype=float64)", "[]", True),
            ("array([10., inf, nan])", "[10. inf nan]", True),
            ("array([-0.5,  inf,  0.2])", "[-0.5  inf  0.2]", True),

            (
                "Conductances (1/V): array([0.1, 0. , nan])",
                "Conductances (1/V): [0.1 0.  nan]",
                True,
            ),
            (
                "Unique voltages: array([1. , 2.5, 4. ])",
                "Unique voltages: [1.  2.5 4. ]",
                True,
            ),
            ("Frequencies: array([3, 3, 1])", "Frequencies: [3 3 1]", True),
            (
                "[ZWCArray([[[1, 2], [2, 0]], [[1, 0], [2, 2]]])]",
                "Target stdout: [array([[[1, 2],\n[2, 0]],\n\n[[1, 0],\n[2, 2]]])]",
                True,
            ),

            (
                "[[np.float64(3.33), np.float64(3.33)], [np.float64(3.33), np.float64(3.33)]]",
                "[[3.33, 3.33], [3.33, 3.33]]",
                True
            )
        ]

        print("Results:")
        for i, (pred, target, expected) in enumerate(test_cases):
            result = self.compare_stdout_outputs(pred, target, self._tolerance)
            status = "✓" if result == expected else "✗"
            print(
                f"{status} Test {i+1}: pred='{pred}' | target='{target}' | Expected: {expected}, Actual: {result}"
            )


if __name__ == "__main__":
    equlityChecker = EqulityChecker(tolerance=1e-2)
    equlityChecker.test_comparison()
