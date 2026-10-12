"""Build the SO-101 Isaac Sim scene from code so it survives restarts.

Run with Isaac Sim's bundled Python (Windows):

    C:\\Users\\recep\\isaacsim\\_build\\windows-x86_64\\release\\python.bat sim\\so101_scene.py
    ... python.bat sim\\so101_scene.py --arms 2 --save sim\\so101_scene.usd --headless

Recreates what was built live through the MCP tools: physics scene, ground plane, and one or
two ``so101_new_calib`` arms (0.4 m apart for two). Written against the ``isaacsim.core.api``
module family (Isaac Sim 4.5 to 6.0).
"""

from __future__ import annotations

import argparse

SO101_ASSET = "/Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd"
ARM_LAYOUTS = {
    1: {"/World/SO101": (0.0, 0.0, 0.0)},
    2: {"/World/SO101_Left": (-0.2, 0.0, 0.0), "/World/SO101_Right": (0.2, 0.0, 0.0)},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SO-101 scene in Isaac Sim.")
    parser.add_argument("--arms", type=int, choices=[1, 2], default=1)
    parser.add_argument("--save", help="Write the stage to this .usd path and exit")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def build_scene(world, arms: int = 1) -> dict:
    """Add ground and SO-101 arms to ``world``. Returns {prim_path: SingleArticulation}."""
    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim asset root not found; check the Nucleus/asset settings.")

    world.scene.add_default_ground_plane()
    articulations = {}
    for prim_path, position in ARM_LAYOUTS[arms].items():
        add_reference_to_stage(usd_path=assets_root + SO101_ASSET, prim_path=prim_path)
        name = prim_path.rsplit("/", 1)[-1].lower()
        articulations[prim_path] = world.scene.add(
            SingleArticulation(prim_path=prim_path, name=name, position=np.array(position))
        )
    return articulations


def joint_limits(articulation) -> dict[str, tuple[float, float]]:
    properties = articulation.dof_properties
    return {
        name: (float(lower), float(upper))
        for name, lower, upper in zip(articulation.dof_names, properties["lower"], properties["upper"])
    }


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    import omni.usd
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0)
    articulations = build_scene(world, arms=args.arms)
    world.reset()
    for prim_path, articulation in articulations.items():
        print(prim_path, articulation.num_dof, "DOF:", articulation.dof_names)

    if args.save:
        omni.usd.get_context().save_as_stage(args.save)
        print(f"Saved stage to {args.save}")
    else:
        while app.is_running():
            world.step(render=True)
    app.close()


if __name__ == "__main__":
    main()
