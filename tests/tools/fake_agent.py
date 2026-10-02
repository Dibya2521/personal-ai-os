"""A stand-in for an outside agent: answers with what it was given.

Prints its arguments, its working directory, any SYNTHIA_ variable it sees
and the task read from stdin; writes "working" to stderr. With FAKE_SLEEP
set it sleeps that many seconds first.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.stderr.write("working\n")
sys.stderr.flush()
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
task = sys.stdin.read()
own = sorted(k for k in os.environ if k.upper().startswith("SYNTHIA_"))
answer = {"args": sys.argv[1:], "cwd": str(Path.cwd()), "own": own, "task": task}
sys.stdout.write(json.dumps(answer) + "\n")
