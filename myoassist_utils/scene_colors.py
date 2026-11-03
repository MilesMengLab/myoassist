"""Brighter palette based on the Unitree MuJoCo G1 scene; rendering only, no control/physics changes.

Source: https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/scene.xml
Brightness and contact force blue are local user preferences.
"""

from functools import lru_cache
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


BLUE = (0.05, 0.3, 1.0, 1.0)


def style_xml(root):
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("diffuse", "0.6 0.6 0.6")
    headlight.set("ambient", "0.45 0.45 0.45")
    headlight.set("specular", "0 0 0")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.4 0.52 0.65 1")
    rgba.set("contactforce", "0.05 0.3 1 1")
    rgba.set("contactfriction", "0.05 0.3 1 1")
    asset = root.find("asset")
    for texture in asset.findall("texture"):
        if texture.get("type") == "skybox":
            texture.attrib.clear()
            texture.attrib.update(type="skybox", builtin="gradient", rgb1="0.5 0.68 0.85",
                                  rgb2="0.3 0.4 0.52", width="512", height="3072")
        elif texture.get("name") == "mount_floor_tex":
            texture.set("builtin", "checker")
            texture.set("rgb1", "0.42 0.52 0.62")
            texture.set("rgb2", "0.3 0.4 0.5")
            texture.set("mark", "edge")
            texture.set("markrgb", "0.8 0.8 0.8")
            texture.set("width", "300")
            texture.set("height", "300")
    for material in asset.findall("material"):
        if material.get("name") == "mount_floor_mat":
            material.set("texuniform", "true")
            material.set("texrepeat", "5 5")
            material.set("reflectance", "0.2")


@lru_cache(maxsize=8)
def _pixels(kind, width, height):
    attrs = (f'type="skybox" builtin="gradient" rgb1="0.5 0.68 0.85" rgb2="0.3 0.4 0.52"'
             if kind == "skybox" else
             'type="2d" builtin="checker" rgb1="0.42 0.52 0.62" rgb2="0.3 0.4 0.5" '
             'mark="edge" markrgb="0.8 0.8 0.8"')
    # Cube/skybox height is six faces, whereas MJCF generator height describes
    # a single face. Preserve the destination buffer layout exactly.
    face_height = height // 6 if kind == "skybox" else height
    temp = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><texture name="theme" {attrs} width="{width}" height="{face_height}"/></asset></mujoco>'
    )
    return temp.tex_data.copy()


def style_model(model):
    """Recolor an archived knee scene in memory; return texture IDs to upload."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mount_preview_floor")
    if floor < 0:
        return []
    model.vis.rgba.haze[:] = [0.4, 0.52, 0.65, 1]
    model.vis.rgba.contactforce[:] = BLUE
    model.vis.rgba.contactfriction[:] = BLUE
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.specular[:] = 0
    material = int(model.geom_matid[floor])
    if material >= 0:
        model.mat_texuniform[material] = 1
        model.mat_texrepeat[material] = [5, 5]
        model.mat_reflectance[material] = 0.2
    changed = []
    for i in range(model.ntex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, i)
        sky = model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX
        if not sky and name != "mount_floor_tex":
            continue
        pixels = _pixels("skybox" if sky else "floor", int(model.tex_width[i]), int(model.tex_height[i]))
        count = int(model.tex_width[i] * model.tex_height[i] * model.tex_nchannel[i])
        if len(pixels) != count:
            raise ValueError("Scene theme texture layout mismatch")
        first = int(model.tex_adr[i])
        model.tex_data[first:first + count] = pixels
        changed.append(i)
    return changed
