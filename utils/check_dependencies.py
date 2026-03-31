#!/usr/bin/env python3
"""Dependency Audit Script.
Scans all Python files in the codebase for imports and compares against requirements.txt
to identify missing dependencies and unused dependencies.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple
import subprocess


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file.
    Side Effects:
        - Reads file from disk
    """
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        # Also catch dynamic imports that might not be in AST
        # Look for patterns like importlib.import_module(), __import__()
        importlib_matches = re.findall(r'importlib\.import_module\([\'"]([^\'"]+)[\'"]', content)
        for match in importlib_matches:
            imports.add(match.split('.')[0])
            
        # Look for __import__ calls
        import_matches = re.findall(r'__import__\([\'"]([^\'"]+)[\'"]', content)
        for match in import_matches:
            imports.add(match.split('.')[0])
            
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    
    return imports


def get_requirements_from_file(file_path: Path) -> Dict[str, str]:
    """Parse requirements.txt and return dict of package -> version.
    Side Effects:
        - Reads file from disk
    """
    requirements = {}
    
    if not file_path.exists():
        print(f"Requirements file {file_path} not found!")
        return requirements
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle various requirement formats
                if '==' in line:
                    package, version = line.split('==', 1)
                elif '>=' in line:
                    package, version = line.split('>=', 1)
                elif '<=' in line:
                    package, version = line.split('<=', 1)
                elif '>' in line:
                    package, version = line.split('>', 1)
                elif '<' in line:
                    package, version = line.split('<', 1)
                else:
                    package, version = line, 'any'
                
                requirements[package.lower().strip()] = version.strip()
    
    return requirements


def is_local_module(module_name: str, project_root: Path) -> bool:
    """Check if a module is a local project module."""
    # List of known local module directories
    local_dirs = {
        'acquire', 'analyze', 'backtest', 'clean', 'common', 'config', 
        'execute', 'model', 'monitor', 'optimize', 'select_assets', 
        'utils', 'tests', 'plots', 'reports', 'links', 'docs', 'sandbox',
        'setups'
    }
    
    # Check if module name matches any local directory (case insensitive)
    if module_name.lower() in local_dirs:
        return True
    
    # Check common local module patterns
    local_patterns = {
        'accountactivity', 'dailydata', 'data_contracts', 'fetchhistory', 
        'get_history', 'get_key_val', 'latest_close', 'logging_setup', 
        'market_cycle', 'portfolio_utils', 'providers', 'pulsyrgain_pct',
        'read_portfolio_positions', 'strategies', 'symbols'
    }
    
    if module_name.lower() in local_patterns:
        return True
    
    # Check if the module file exists in the project
    module_file = project_root / f"{module_name}.py"
    if module_file.exists():
        return True
    
    # Check if it's a subdirectory with __init__.py
    module_dir = project_root / module_name
    if module_dir.is_dir() and (module_dir / "__init__.py").exists():
        return True
    
    return False


def map_import_to_package(import_name: str, project_root: Path) -> str:
    """Map Python import names to pip package names."""
    # First check if it's a local module
    if is_local_module(import_name, project_root):
        return "local_module"
    
    # Common mappings for external packages
    import_to_package = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'bs4': 'beautifulsoup4',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'requests': 'requests',
        'yfinance': 'yfinance',
        'pyarrow': 'pyarrow',
        'fastparquet': 'fastparquet',
        'pytest': 'pytest',
        'coverage': 'coverage',
        'tqdm': 'tqdm',
        'exchange_calendars': 'exchange_calendars',
        'trendet': 'trendet',
        'seaborn': 'seaborn',
        'selenium': 'selenium',
        'polars': 'polars',
        'backtrader': 'backtrader',
        'alpaca': 'alpaca-trade-api',
        'setuptools': 'setuptools',
        'pytz': 'pytz',
        'zoneinfo': 'zoneinfo',  # Built-in in Python 3.9+
    }
    
    return import_to_package.get(import_name, import_name)


def is_builtin_module(module_name: str) -> bool:
    """Check if a module is a Python built-in."""
    builtins = {
        # Core Python built-ins
        '__future__', 'abc', 'ast', 'atexit', 'builtins', 'collections', 'contextlib',
        'copy', 'copyreg', 'csv', 'dataclasses', 'datetime', 'decimal', 'difflib',
        'email', 'enum', 'fractions', 'functools', 'gc', 'genericpath', 'glob',
        'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib', 'inspect', 'io',
        'ipaddress', 'json', 'linecache', 'locale', 'logging', 'marshal', 'math',
        'mimetypes', 'multiprocessing', 'netrc', 'nntplib', 'numbers', 'operator',
        'os', 'pathlib', 'pickle', 'pickletools', 'pkgutil', 'platform', 'plistlib',
        'poplib', 'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'pyclbr',
        'pydoc', 'queue', 'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter',
        'runpy', 'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil',
        'signal', 'site', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'sqlite3',
        'sre', 'sre_compile', 'sre_constants', 'sre_parse', 'ssl', 'stat', 'statistics',
        'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'sys', 'sysconfig',
        'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'textwrap',
        'this', 'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace',
        'traceback', 'tracemalloc', 'tty', 'types', 'typing', 'unicodedata', 'unittest',
        'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
        'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile',
        'zipimport', 'zoneinfo', 'zlib', 'configparser', 'argparse', 'itertools',
        'collections.abc', 'typing_extensions', 'concurrent', 'concurrent.futures',
        'asyncio', 'asyncore', 'asynchat', 'binascii', 'base64', 'bisect', 'bz2',
        'code', 'codecs', 'codeop', 'colorsys', 'compileall', 'contextvars', 'cProfile',
        'crypt', 'ctypes', 'datetime', 'dbm', 'dis', 'distutils', 'doctest', 'email',
        'ensurepip', 'errno', 'faulthandler', 'filecmp', 'fileinput', 'fnmatch',
        'formatter', 'fpectl', 'ftplib', 'gc', 'getopt', 'getpass', 'gettext', 'grp',
        'gzip', 'hashlib', 'heapq', 'hmac', 'imaplib', 'imghdr', 'imp', 'importlib',
        'inspect', 'io', 'ipaddress', 'json', 'lib2to3', 'linecache', 'locale',
        'lzma', 'mailbox', 'mailcap', 'marshal', 'mimetypes', 'modulefinder',
        'multiprocessing', 'netrc', 'nntplib', 'ntpath', 'nturl2path', 'numbers',
        'opcode', 'optparse', 'os', 'ossaudiodev', 'pathlib', 'pdb', 'pickle',
        'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
        'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile',
        'pyclbr', 'pydoc', 'pydoc_data', 'pyexpat', 'queue', 'quopri', 'random',
        're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
        'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
        'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'sqlite3',
        'sre', 'sre_compile', 'sre_constants', 'sre_parse', 'ssl', 'stat', 'statistics',
        'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symbol', 'symtable',
        'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile',
        'termios', 'textwrap', 'this', 'threading', 'time', 'timeit', 'tkinter',
        'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle',
        'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid',
        'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
        'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport',
        'zoneinfo', 'zlib'
    }
    return module_name in builtins


def scan_python_files(project_root: Path) -> Dict[Path, Set[str]]:
    """Scan all Python files in the project and extract imports."""
    all_imports = {}
    
    # Skip certain directories
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'env'}
    
    for py_file in project_root.rglob('*.py'):
        # Skip files in directories we want to ignore
        if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
            continue
            
        imports = extract_imports_from_file(py_file)
        if imports:
            all_imports[py_file] = imports
    
    return all_imports


def main() -> int:
    """Main dependency audit function."""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / 'requirements.txt'
    
    print("DEPENDENCY AUDIT")
    print("=" * 50)
    
    # Scan all Python files
    print("Scanning Python files...")
    all_imports = scan_python_files(project_root)
    
    # Get all unique imports
    all_unique_imports = set()
    for imports in all_imports.values():
        all_unique_imports.update(imports)
    
    print(f"   Found {len(all_imports)} Python files")
    print(f"   Found {len(all_unique_imports)} unique imports")
    
    # Get current requirements
    print("\nReading requirements.txt...")
    requirements = get_requirements_from_file(requirements_file)
    print(f"   Found {len(requirements)} requirements")
    
    # Map imports to packages and filter out built-ins and local modules
    external_packages = set()
    for import_name in all_unique_imports:
        if not is_builtin_module(import_name):
            package_name = map_import_to_package(import_name, project_root)
            if package_name != "local_module":
                external_packages.add(package_name.lower())
    
    # Find missing dependencies
    print("\nMISSING DEPENDENCIES:")
    missing = external_packages - set(requirements.keys())
    if missing:
        for package in sorted(missing):
            print(f"   {package}")
    else:
        print("   No missing dependencies found!")
    
    # Find potentially unused dependencies
    print("\nPOTENTIALLY UNUSED DEPENDENCIES:")
    unused = set(requirements.keys()) - external_packages
    if unused:
        for package in sorted(unused):
            print(f"   {package} ({requirements[package]})")
    else:
        print("   No unused dependencies found!")
    
    # Generate updated requirements.txt
    if missing:
        print("\nSUGGESTED UPDATES TO requirements.txt:")
        print("   Add these packages:")
        for package in sorted(missing):
            print(f"   {package}")
        
        # Create updated requirements file
        updated_requirements = list(requirements.items())
        for package in sorted(missing):
            updated_requirements.append((package, 'any'))
        
        updated_requirements.sort(key=lambda x: x[0])
        
        print(f"\nUpdated requirements.txt would contain {len(updated_requirements)} packages")
        print("To update requirements.txt, run:")
        print("   python utils/check_dependencies.py --update")
    else:
        print("\nAll dependencies are properly listed!")
    
    print(f"\nAudit complete!")
    return len(missing)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
