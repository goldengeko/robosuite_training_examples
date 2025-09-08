import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Kinova3_6DOF(ManipulatorModel):
    """
    The Gen3 robot is the sparkly newest addition to the Kinova line

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/kinova3_6dof/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_gripper(self):
        return {"right": "Robotiq140Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_kinova3_6dof"}

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, 1.890, 0.000, 0.600])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
