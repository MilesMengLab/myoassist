"""Audit upper-hinge fitting against actual human bone mesh surfaces.

Reports upper brace clearance separately from fixed motor/shank geometry. No skin
or strap-pressure model, and finite pose samples do not certify all motion.
"""

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.check_knee_mount_clearance import mesh_geometry, surface_crossings, surface_distance
from tools.preview_knee_hinge_mount import DEFAULT_CONFIG, DEFAULT_OUTPUT, LINKS, build, load_config, update


def minimum_clearance(model, data, geom, human_geoms):
    first = mesh_geometry(model, data, geom)
    rows = []
    for human in human_geoms:
        second = mesh_geometry(model, data, human)
        # An AABB gap is a lower bound. Sort these so distant anatomy needs no
        # expensive triangle query once a closer surface has been found.
        gap = np.maximum(0, np.maximum(first[0].min(0) - second[0].max(0),
                                      second[0].min(0) - first[0].max(0)))
        rows.append((float(np.linalg.norm(gap)), human, second))
    best = {"distance_mm": float("inf"), "human_geom": None, "crossing_edges": 0}
    for lower_bound, human, second in sorted(rows, key=lambda r: r[0]):
        if lower_bound * 1000 > best["distance_mm"]:
            continue
        crossings = surface_crossings(second, first)
        distance = 0. if crossings else surface_distance(second, first) * 1000
        if distance < best["distance_mm"]:
            best = {"distance_mm": distance, "human_geom": model.geom(human).name,
                    "crossing_edges": int(crossings)}
    return best


def audit(config_path=DEFAULT_CONFIG, output=DEFAULT_OUTPUT, angles=(0, 30, 60, 90)):
    output = Path(output)
    model, data, devices = build(output)
    config = load_config(config_path)
    neutral = copy.deepcopy(config)
    for side in neutral["sides"]:
        neutral["sides"][side]["upper_hinge_deg"] = 0
    human = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
             and not model.geom(g).name.startswith("hinge_")]
    target_mm = config.get("target_clearance_mm", 5.0)
    tolerance_mm = config.get("clearance_tolerance_mm", .05)
    report = {"config": config, "target_clearance_mm": target_mm,
              "clearance_tolerance_mm": tolerance_mm, "poses": [], "scope":
              "Triangle mesh surfaces of upper base_link versus all rendered bones. No skin, muscle-line or pressure clearance. Fixed motor/output/shank are not subject to the upper-brace target."}
    for angle in angles:
        update(model, data, devices, neutral, angle)
        before_pos, before_rot = data.xpos.copy(), data.xmat.copy()
        alignment = update(model, data, devices, config, angle)
        row = {"flexion_deg": angle, "alignment": alignment, "sides": {}}
        for side in ("R", "L"):
            fixed = {}
            for link in LINKS[1:]:
                body = model.body(f"hinge_{side}_{link}").id
                fixed[link] = {
                    "translation_change_mm": float(np.linalg.norm(data.xpos[body] - before_pos[body]) * 1000),
                    "rotation_matrix_max_change": float(np.abs(data.xmat[body] - before_rot[body]).max()),
                }
            geom = model.geom(f"hinge_{side}_base_link_visual_0").id
            clearance = minimum_clearance(model, data, geom, human)
            row["sides"][side] = {"upper_brace": clearance, "fixed_parts": fixed}
        report["poses"].append(row)
        print(json.dumps(row), flush=True)
    report["all_sampled_upper_brace_gaps_at_target"] = all(
        abs(row["sides"][s]["upper_brace"]["distance_mm"] - target_mm) <= tolerance_mm
        and row["sides"][s]["upper_brace"]["crossing_edges"] == 0
        for row in report["poses"] for s in ("R", "L")
    )
    target = output / "clearance_report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit(args.config, args.output)


if __name__ == "__main__":
    main()
