# Do not touch this GUI_Mananger.py file.
import os
import sys
from pathlib import Path

# [✔ OS Compatibility Setup]
# For Ubuntu 22.04 users:
# → Uncomment the following line,
# → And comment out the line below that points to "bin".
# sys.path.append(str(Path(__file__).resolve().parent / "bin_for_ubuntu22.04"))
sys.path.append(str(Path(__file__).resolve().parent / "bin"))  # Default: for Ubuntu 24.04 and above

current_dir = Path(__file__).resolve().parent
os.environ["QT_PLUGIN_PATH"] = current_dir.as_posix()
import ai_bmt_inteface_python as bmt

def ExecuteGUI(global_interface, model_path):
    global_caller = bmt.AI_BMT_GUI_CALLER(global_interface, model_path)
    args = [sys.argv[0], "--current_dir", current_dir.as_posix()]
    global_caller.call_BMT_GUI(args)
    return 0

