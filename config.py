import os
import pwd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
PROJECT_NAME = "ResearchNLP"
CODE_DIR = os.path.join(ROOT_DIR, PROJECT_NAME) + "/"
expr_log_dir_abspath = os.path.join(CODE_DIR, "experiments_files/")
tmp_expr_folder_prefix = "/tmp/ResearchNLP/"
is_server = pwd.getpwuid(os.getuid()).pw_name == 'yonatanz'  # server comp