import json
import socket
from typing import Optional, Tuple

from mcp.server.fastmcp import FastMCP

robot_host = "127.0.0.1"
robot_port = "5000"


class RealmanChassis:
    """
    A client class for communicating with the Realman robot chassis over TCP.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize the RealmanChassis client.

        Args:
            host (str): IP address or hostname of the chassis server.
            port (int): Port number to connect to.

        Raises:
            ConnectionError: If the socket connection fails.
        """
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client.connect((self.host, self.port))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")

    def get_current_pose(self) -> Optional[dict]:
        """
        Query the current pose of the robot.

        Returns:
            dict or None: The current pose dictionary if successful, else None.
        """
        try:
            self.client.sendall(b"/api/robot_status")
            response = self.client.recv(2048).decode()
            data = json.loads(response)
            return data.get("results", {}).get("current_pose")
        except Exception as e:
            print(f"[ERROR] Failed to get current pose: {e}")
            return None

    def move_to_position(
        self, marker_name: str
    ) -> Tuple[Optional[dict], Optional[dict], str]:
        """
        Move the robot to a specified marker position.

        Args:
            marker_name (str): The name of the marker to move to.

        Returns:
            Tuple[Optional[dict], Optional[dict], str]:
                - start_pose: Pose before the move
                - target_pose: Pose after the move
                - status: Move status string (e.g., "success", "failed")
        """
        start_pose = self.get_current_pose()
        request_move = f"/api/move?marker={marker_name}"
        self.client.sendall(request_move.encode("utf-8"))
        response_move = self.client.recv(2048).decode()
        try:
            data_move = json.loads(response_move)
        except json.JSONDecodeError:
            return start_pose, None, "Failed to parse move response"

        # Wait for move to complete (code "01002" indicates completion)
        while True:
            try:
                response_status = self.client.recv(2048).decode()
                data_status = json.loads(response_status)
                if data_status.get("code") == "01002":
                    break
            except json.JSONDecodeError:
                print(f"[WARN] Ignoring invalid JSON chunk.")
                continue

        target_pose = self.get_current_pose()
        return start_pose, target_pose, data_move.get("status", "unknown")

    def cancel_current_move(self):
        """
        Cancel the current move command if in progress.
        """
        try:
            self.client.sendall(b"/api/move/cancel")
            response = self.client.recv(2048).decode()
            print(f"[INFO] Cancel move response: {response}")
        except Exception as e:
            print(f"[ERROR] Cancel move failed: {e}")

    def close_connection(self):
        """
        Close the socket connection to the chassis server.
        """
        self.client.close()
        print(f"[INFO] Connection closed")


# Initialize FastMCP server
mcp = FastMCP("robots")


@mcp.tool()
def navigate_to_target(marker_name: str) -> dict:
    """
    Perform navigation of the robot to a specified marker position.

    Args:
        marker_name (str): The destination marker name.

    Returns:
        str: A message indicating whether navigation to the marker succeeded or failed.

    Raises:
        FileNotFoundError: If `config.yaml` does not exist.
        KeyError: If required keys (host, port) are missing in the config file.
        Exception: Any error during robot communication or movement execution.

    """

    robot = RealmanChassis(robot_host, robot_port)
    robot.cancel_current_move()
    _, _, status = robot.move_to_position(marker_name)
    robot.close_connection()

    return (
        f"Navigation to '{marker_name}' {'succeeded' if status == 'ok' else 'failed'}"
    )


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
