import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import inspect


class DocstringCollector:
    """A utility class for collecting and organizing docstrings from Python source files."""

    def __init__(self):
        self.collected_info = {}  # Stores all collected documentation info
        self.current_module_imports = set()  # Track imports in current module

    def get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract the function signature from an AST node."""
        args = []

        # Process arguments
        for arg in node.args.args:
            arg_name = arg.arg
            # Get annotation if it exists
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    # Handle basic generic types like List[str]
                    arg_type = self._format_annotation(arg.annotation)
                else:
                    arg_type = "Any"
                args.append(f"{arg_name}: {arg_type}")
            else:
                args.append(arg_name)

        # Add *args if present
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # Add **kwargs if present
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # Get return annotation if it exists
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = f" -> {node.returns.id}"
            elif isinstance(node.returns, ast.Subscript):
                return_type = f" -> {self._format_annotation(node.returns)}"
            else:
                return_type = " -> Any"
        else:
            return_type = ""

        return f"def {node.name}({', '.join(args)}){return_type}"

    def _format_annotation(self, node: ast.Subscript) -> str:
        """Format type annotation nodes into string representation."""
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if isinstance(node.slice, ast.Name):
                param = node.slice.id
            elif isinstance(node.slice, ast.Tuple):
                param = ", ".join(
                    elt.id for elt in node.slice.elts if isinstance(elt, ast.Name)
                )
            else:
                param = "Any"
            return f"{base}[{param}]"
        return "Any"

    def collect_imports(self, node: ast.AST) -> None:
        """Collect import statements from the module."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.Import):
                    for name in child.names:
                        self.current_module_imports.add(name.name)
                else:
                    module = child.module if child.module else ""
                    for name in child.names:
                        self.current_module_imports.add(f"{module}.{name.name}")

    def extract_info(
        self, node: ast.AST, module_name: str, parent_name: Optional[str] = None
    ) -> None:
        """Recursively extract documentation info from an AST node."""
        # Get the qualified name for this node
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if parent_name:
                qualified_name = f"{parent_name}.{node.name}"
            else:
                qualified_name = node.name
        else:
            qualified_name = parent_name if parent_name else "module"

        # Extract documentation info
        docstring = ast.get_docstring(node)
        if docstring:
            if module_name not in self.collected_info:
                self.collected_info[module_name] = {
                    "imports": self.current_module_imports,
                    "items": [],
                }

            item_info = {
                "name": qualified_name,
                "type": type(node).__name__,
                "docstring": docstring,
            }

            # Add function signature if it's a function
            if isinstance(node, ast.FunctionDef):
                item_info["signature"] = self.get_function_signature(node)

            self.collected_info[module_name]["items"].append(item_info)

        # Recursively process child nodes
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                self.extract_info(child, module_name, qualified_name)

    def process_file(self, file_path: str) -> None:
        """Process a single Python file and extract its documentation info."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
            self.current_module_imports = set()  # Reset imports for new module
            self.collect_imports(tree)  # Collect imports

            rel_path = str(Path(file_path))
            module_name = rel_path.replace(".py", "").replace("/", ".")
            self.extract_info(tree, module_name)

        except SyntaxError as e:
            print(f"Failed to parse {file_path}: {str(e)}")

    def process_directory(self, directory: str, recursive: bool = False) -> None:
        """Process all Python files in a directory."""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.process_file(file_path)
            if not recursive:
                break

    def write_markdown(self, output_file: str) -> None:
        """Write collected documentation to a markdown file."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# API Documentation\n\n")
            f.write("## Table of Contents\n\n")

            # Generate TOC
            for module_path in sorted(self.collected_info.keys()):
                module_ref = module_path.replace(".", "-").lower()
                f.write(f"- [{module_path}](#{module_ref})\n")
            f.write("\n---\n\n")

            # Write detailed documentation
            for module_path, module_info in sorted(self.collected_info.items()):
                f.write(f"## {module_path}\n\n")

                # Write module-level docstring first if it exists
                module_docs = [d for d in module_info["items"] if d["name"] == "module"]
                if module_docs:
                    f.write(f"{module_docs[0]['docstring']}\n\n")

                # Write class documentation
                classes = [d for d in module_info["items"] if d["type"] == "ClassDef"]
                if classes:
                    f.write("### Classes\n\n")
                    for class_doc in classes:
                        f.write(
                            f"#### {class_doc['name']}\n\n{class_doc['docstring']}\n\n"
                        )

                        # Find methods belonging to this class
                        methods = [
                            d
                            for d in module_info["items"]
                            if d["type"] == "FunctionDef"
                            and d["name"].startswith(f"{class_doc['name']}.")
                        ]
                        if methods:
                            f.write("##### Methods\n\n")
                            for method in methods:
                                f.write(f"```python\n{method['signature']}\n```\n\n")
                                f.write(f"{method['docstring']}\n\n")

                # Write standalone function documentation
                functions = [
                    d
                    for d in module_info["items"]
                    if d["type"] == "FunctionDef" and "." not in d["name"]
                ]
                if functions:
                    f.write("### Functions\n\n")
                    for func_doc in functions:
                        f.write(f"```python\n{func_doc['signature']}\n```\n\n")
                        f.write(f"{func_doc['docstring']}\n\n")

                f.write("---\n\n")


def main():
    """Main function to run the documentation collector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate API documentation from Python source files"
    )
    parser.add_argument("--input_path", help="Path to Python file or directory")
    parser.add_argument("--output_file", help="Path for output markdown file")
    parser.add_argument(
        "--recursive", action="store_true", help="Process files in subdirectories"
    )

    args = parser.parse_args()

    collector = DocstringCollector()

    if os.path.isfile(args.input_path):
        collector.process_file(args.input_path)
    else:
        collector.process_directory(args.input_path, args.recursive)

    collector.write_markdown(args.output_file)
    print(f"Documentation written to {args.output_file}")


if __name__ == "__main__":
    main()
