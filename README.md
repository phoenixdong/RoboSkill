# Objective
Leverage MCP's standardization capabilities to provide a unified vendor integration specification, creating an embodied robot universal skill store that enables one-click integration of robot skills with robot frameworks.

1. Robot skills are registered according to the MCP specification.
2. Organized in a hierarchical structure of `Manufacturer Name -> Model Name`.

# Usage
Best used in conjunction with [FlagOpen/RoboOS](https://github.com/FlagOpen/RoboOS)

# Vendor Adaptation Process
1. Create a manufacturer directory
2. Create a robot model directory
3. Develop skill functions based on MCP specifications

# Local Testing
Take `demo_manufacturer/demo_model` as an example:
```bash
cd demo_manufacturer/demo_model
mcp dev skill.py