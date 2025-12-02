# disable_zfpy.py
import sys
import os
os.environ.setdefault('NUMCODECS_DISABLE_ZFPY', '1')
sys.modules.setdefault('numcodecs.zfpy', None)
sys.modules.setdefault('zfpy', None)
