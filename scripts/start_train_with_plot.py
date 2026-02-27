import sys
import argparse

from PyQt5 import QtWidgets

# from evaluate_td3 import evaluate
from utils.thread_train import TrainingThread
# from utils.thread_train_fixedwing import TrainingThread
from utils.ui_train import TrainingUi
from configparser import ConfigParser


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training navigation model using TD3")
    parser.add_argument('-config', required=True,
                        help='config file name, such as config0925.ini', default='config_default.ini')

    return parser


def main():
    args = get_parser().parse_args()

    # Accept:
    # 1) config_Blocks_Multirotor_2D.ini
    # 2) configs/config_Blocks_Multirotor_2D.ini
    config_arg = args.config
    if not config_arg.endswith(".ini"):
        raise ValueError("config must end with .ini")
    if not config_arg.startswith("configs/") and not config_arg.startswith("configs\\"):
        config_file = f"configs/{config_arg}"
    else:
        config_file = config_arg

    # 1. Create the qt thread
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()

    # 2. Start training thread
    training_thread = TrainingThread(config_file)

    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)

    # Automatically close the GUI when the training thread finishes
    training_thread.finished.connect(app.quit)

    cfg = ConfigParser()
    cfg.read(config_file)
    training_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
