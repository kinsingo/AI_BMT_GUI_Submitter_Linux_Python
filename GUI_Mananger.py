# Do not touch this GUI_Mananger.py file.
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "bin"))
current_dir = Path(__file__).resolve().parent
os.environ["QT_PLUGIN_PATH"] = current_dir.as_posix()
import ai_bmt_inteface_python as bmt

def ExecuteGUI(global_interface, model_path):
    global_caller = bmt.AI_BMT_GUI_CALLER(global_interface, model_path)
    args = [sys.argv[0], "--current_dir", current_dir.as_posix()]
    global_caller.call_BMT_GUI(args)
    return 0

