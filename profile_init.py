# profile_init.py
import os

# turn on profiling mode BEFORE importing __main__
os.environ["SASPRO_STARTUP_PROFILE"] = "1"

from setiastro.saspro.__main__ import main

if __name__ == "__main__":
    main()
