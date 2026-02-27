import argparse
import math
import time

import airsim
import keyboard


def get_yaw_rad(client: airsim.MultirotorClient) -> float:
    q = client.simGetVehiclePose().orientation
    _, _, yaw = airsim.to_eularian_angles(q)
    return yaw


def body_to_world(vx_body: float, vy_body: float, yaw_rad: float) -> tuple[float, float]:
    vx_world = vx_body * math.cos(yaw_rad) - vy_body * math.sin(yaw_rad)
    vy_world = vx_body * math.sin(yaw_rad) + vy_body * math.cos(yaw_rad)
    return vx_world, vy_world


def main() -> None:
    parser = argparse.ArgumentParser(description="AirSim Multirotor WASD realtime manual control")
    parser.add_argument("--speed", type=float, default=2.0, help="horizontal speed (m/s)")
    parser.add_argument("--yaw-rate", type=float, default=30.0, help="yaw rate command (deg/s)")
    parser.add_argument("--climb-rate", type=float, default=1.0, help="vertical setpoint change speed (m/s)")
    parser.add_argument("--dt", type=float, default=0.05, help="control loop period (s)")
    parser.add_argument("--takeoff-height", type=float, default=3.0, help="takeoff target height (m)")
    parser.add_argument("--no-takeoff", action="store_true", help="do not auto takeoff")
    parser.add_argument("--no-land", action="store_true", help="do not auto land on exit")
    args = parser.parse_args()

    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    if not args.no_takeoff:
        print(f"Taking off and moving to {args.takeoff_height:.1f}m...")
        client.takeoffAsync().join()
        z_setpoint = -abs(args.takeoff_height)
        client.moveToZAsync(z_setpoint, 1.0).join()
    else:
        z_setpoint = client.simGetVehiclePose().position.z_val

    print("Manual control started.")
    print("W/S: forward/back | A/D: left/right | Q/E: yaw left/right")
    print("R/F: up/down | ESC: exit")
    print("If yaw direction feels opposite in your scene, swap Q/E mapping in script.")

    try:
        while True:
            if keyboard.is_pressed("esc"):
                break

            vx_body = 0.0
            vy_body = 0.0
            yaw_rate_cmd = 0.0

            if keyboard.is_pressed("w"):
                vx_body += args.speed
            if keyboard.is_pressed("s"):
                vx_body -= args.speed
            if keyboard.is_pressed("a"):
                vy_body -= args.speed
            if keyboard.is_pressed("d"):
                vy_body += args.speed

            if keyboard.is_pressed("q"):
                yaw_rate_cmd -= abs(args.yaw_rate)
            if keyboard.is_pressed("e"):
                yaw_rate_cmd += abs(args.yaw_rate)

            if keyboard.is_pressed("r"):
                z_setpoint -= abs(args.climb_rate) * args.dt
            if keyboard.is_pressed("f"):
                z_setpoint += abs(args.climb_rate) * args.dt

            yaw = get_yaw_rad(client)
            vx_world, vy_world = body_to_world(vx_body, vy_body, yaw)

            client.moveByVelocityZAsync(
                vx_world,
                vy_world,
                z_setpoint,
                args.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_cmd),
            )
            time.sleep(args.dt)

    finally:
        print("Stopping...")
        client.moveByVelocityZAsync(0, 0, z_setpoint, 0.2).join()
        if not args.no_land:
            print("Landing...")
            client.landAsync().join()
            client.armDisarm(False)
        client.enableApiControl(False)
        print("Exited.")


if __name__ == "__main__":
    main()
