import atexit
import os
import sys

from minecraft_v.progress import phase

with phase("[load] starting"):
    from minecraft_v.main import main

main()
# Done: skip freeing every block and cell one by one on the way out, which takes seconds on a big build.
atexit._run_exitfuncs()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
