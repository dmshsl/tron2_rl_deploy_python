import atexit
import time

import numpy as np
import yaml


class DryRunRobot:
    """Robot 代理：拦截 publishRobotCmd，只校验并打印关节指令，绝不下发到硬件。"""

    def __init__(self, robot, config_file, print_interval=0.5):
        self._robot = robot
        self._print_interval = float(print_interval)

        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)['PointfootCfg']

        control = cfg['control']
        self._joint_names = cfg['init_state']['joint_names']
        joint_num = len(self._joint_names)
        self._modes = control.get('joint_control_modes', ['position'] * joint_num)

        leg_limit = float(control.get('leg_joint_torque_limit', control.get('user_torque_limit', 0.0)))
        light_limit = float(control.get('light_joint_torque_limit', leg_limit))
        wheel_limit = float(control.get('wheel_joint_torque_limit', 0.0))
        # 关节 3/8（(i+1)%5==3）是 proximal_yaw，控制器按 light_joint_* 增益驱动
        self._torque_limits = np.array([
            wheel_limit if self._modes[i] == 'velocity'
            else light_limit if (i + 1) % 5 == 3
            else leg_limit
            for i in range(joint_num)
        ])

        self._state_q = np.zeros(joint_num)
        self._state_dq = np.zeros(joint_num)
        self._state_time = 0.0
        self._cmd_count = 0
        self._start_time = time.time()
        self._last_print = 0.0
        self._peak_tau = np.zeros(joint_num)
        self._violations = np.zeros(joint_num, dtype=int)

        atexit.register(self.print_summary)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_robot'), name)

    def subscribeRobotState(self, callback):
        # 包一层而不是另起订阅，避免 SDK 单回调实现把控制器的回调覆盖掉
        def wrapper(state):
            self._state_q = np.asarray(state.q, dtype=float)
            self._state_dq = np.asarray(state.dq, dtype=float)
            self._state_time = time.time()
            callback(state)

        return self._robot.subscribeRobotState(wrapper)

    def publishRobotCmd(self, cmd):
        q_des = np.asarray(cmd.q, dtype=float)
        dq_des = np.asarray(cmd.dq, dtype=float)
        tau_ff = np.asarray(cmd.tau, dtype=float)
        kp = np.asarray(cmd.Kp, dtype=float)
        kd = np.asarray(cmd.Kd, dtype=float)

        # 驱动器实际执行的是 PD + 前馈，这才是关节真正受到的力矩
        tau_est = kp * (q_des - self._state_q) + kd * (dq_des - self._state_dq) + tau_ff

        self._cmd_count += 1
        self._peak_tau = np.maximum(self._peak_tau, np.abs(tau_est))
        self._violations += (np.abs(tau_est) > self._torque_limits).astype(int)

        now = time.time()
        if now - self._last_print >= self._print_interval:
            self._last_print = now
            self._print_table(q_des, dq_des, tau_ff, kp, kd, tau_est, now)
        return True

    def _print_table(self, q_des, dq_des, tau_ff, kp, kd, tau_est, now):
        if self._state_time == 0.0:
            state_info = "\033[31mNO ROBOT STATE — q/dq 全为 0，tau_est 不可信\033[0m"
        else:
            state_info = f"state age {(now - self._state_time) * 1000:.0f} ms"
        rate = self._cmd_count / max(now - self._start_time, 1e-6)

        print("=" * 108)
        print(f"[DRY-RUN] cmds={self._cmd_count}  {rate:.1f} Hz  {state_info}")
        print(f"{'idx':>3} {'joint':<17}{'mode':<5}{'q_des':>9}{'q_now':>9}{'dq_des':>9}"
              f"{'dq_now':>9}{'Kp':>9}{'Kd':>8}{'tau_est':>10}{'limit':>8}  flag")
        for i, name in enumerate(self._joint_names):
            bad = abs(tau_est[i]) > self._torque_limits[i] or not np.isfinite(tau_est[i])
            flag = "\033[31m OVER\033[0m" if bad else ""
            mode = "vel" if self._modes[i] == 'velocity' else "pos"
            print(f"{i:>3} {name:<17}{mode:<5}{q_des[i]:>9.4f}{self._state_q[i]:>9.4f}"
                  f"{dq_des[i]:>9.3f}{self._state_dq[i]:>9.3f}{kp[i]:>9.2f}{kd[i]:>8.2f}"
                  f"{tau_est[i]:>10.2f}{self._torque_limits[i]:>8.1f} {flag}")
        if np.any(tau_ff != 0.0):
            print(f"    tau_ff (前馈力矩): {np.array2string(tau_ff, precision=2)}")

    def print_summary(self):
        if self._cmd_count == 0:
            return
        print("\n" + "=" * 108)
        print(f"[DRY-RUN] 汇总：共拦截 {self._cmd_count} 条指令，未向机器人下发任何数据")
        print(f"{'idx':>3} {'joint':<17}{'peak|tau_est|':>15}{'limit':>10}{'over-count':>12}")
        for i, name in enumerate(self._joint_names):
            over = self._violations[i]
            color = "\033[31m" if over else ""
            reset = "\033[0m" if over else ""
            print(f"{i:>3} {name:<17}{color}{self._peak_tau[i]:>15.2f}{reset}"
                  f"{self._torque_limits[i]:>10.1f}{over:>12}")
        if self._state_time == 0.0:
            print("\033[31m注意：全程未收到 RobotState，以上力矩估算基于 q=0/dq=0，仅能验证指令数值本身。\033[0m")
        print("=" * 108)
