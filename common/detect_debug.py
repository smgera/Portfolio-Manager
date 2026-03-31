import sys
print("debug" if sys.gettrace() or 'debugpy' in sys.modules else "not debug")

if not sys.gettrace() and not 'debugpy' in sys.modules:
    print("not debug")
else:
    print("debug")
