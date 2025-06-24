from navigate.navigation import RealmanChassis

def navigate_to_target(host: str, port: int, marker_name: str) -> dict:
    """
    Perform navigation of the robot to a specified marker position.

    Args:
        host (str): IP address or hostname of the robot.
        port (int): TCP port used for communication.
        marker_name (str): The destination marker name.

    Returns:
        dict: Result dictionary containing success status, message, 
              and optional pose data (start_pose, target_pose).
    """
    try:
        robot = RealmanChassis(host, port)
        robot.cancel_current_move()  # Ensure any previous motion is canceled
        _, _, status = robot.move_to_position(marker_name)
        robot.close_connection()

        return {
            "success": status == "ok",
            "message": f"Navigation to '{marker_name}' {'succeeded' if status == 'ok' else 'failed'}"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Navigation to '{marker_name}' failed: {str(e)}"
        }


if __name__ == '__main__':
    # Example usage
    result = navigate_to_target("192.168.10.10", 31001, "p0")
    print(result)
