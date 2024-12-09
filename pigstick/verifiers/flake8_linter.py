from flake8.api import legacy as flake8
import tempfile

def lint_python_code(code_string):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code_string)
        temp_file.flush()
        style_guide = flake8.get_style_guide()
        report = style_guide.check_files([temp_file.name])
        return report.get_statistics('E'), report.get_statistics('W'), report.get_statistics('F')


if __name__ == "__main__":
    code_string = """
def foo()
    print("Hello, World!")
"""
    errors, warnings, failures = lint_python_code(code_string)
    print("Errors:", errors)
    print("Warnings:", warnings)
    print("Failures:", failures)
