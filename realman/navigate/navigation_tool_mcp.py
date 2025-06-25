import yaml
from mcp.server.fastmcp import FastMCP

from navigation import RealmanChassis

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

    config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    host = config["robot"]["host"]
    port = config["robot"]["port"]
    robot = RealmanChassis(host, port)
    robot.cancel_current_move()
    _, _, status = robot.move_to_position(marker_name)
    robot.close_connection()

    return (
        f"Navigation to '{marker_name}' {'succeeded' if status == 'ok' else 'failed'}"
    )


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
