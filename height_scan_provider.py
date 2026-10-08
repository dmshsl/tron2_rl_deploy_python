"""Height provider boundary for simulator GT or timestamped camera perception."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from tron2_deploy.perception.height_map import DepthFrame, HeightMap
from tron2_deploy.perception.odometry import Odometry, OdometryInput
from tron2_deploy.policy_contract import DeploymentContract
from tron2_deploy.policy_observation import SdkState, Vector


class HeightScanProvider(Protocol):
    def get_scan(self, state: SdkState) -> Vector:
        """Return 231 x-fastest geometric values, before policy term scale/clip."""
        ...


class PerceptionHeightScanProvider:
    """Wrap the existing odometry/map with an injected synchronized depth source.

    A source returns at most one front/rear capture, using state.timestamp's clock.
    Camera drivers and simulator renderers supply the same DepthFrame API.
    """

    def __init__(self, contract: DeploymentContract,
                 frame_source: Callable[[float], tuple[DepthFrame, ...]]) -> None:
        self.odometry = Odometry(imu_to_base=Rotation.from_euler("xyz", contract.imu.imu_to_base_rpy).as_matrix())
        self.height_map = HeightMap()
        self.frame_source = frame_source

    def get_scan(self, state: SdkState) -> Vector:
        pose = self.odometry.update(OdometryInput(state.timestamp, state.quat.astype(np.float64),
                                                state.gyro.astype(np.float64), state.q.astype(np.float64),
                                                state.dq.astype(np.float64)))
        result = self.height_map.update(self.frame_source(state.timestamp), pose.pose, state.timestamp)
        return result.values.astype(np.float32)
