# disable_zfpy.py
import sys, os
# Make any attempt to import the zfpy extension fail-fast to the pure-python path.
os.environ.setdefault('NUMCODECS_DISABLE_ZFPY', '1')
# Also poison the module names so import falls back or raises cleanly.
sys.modules.setdefault('numcodecs.zfpy', None)
sys.modules.setdefault('zfpy', None)
