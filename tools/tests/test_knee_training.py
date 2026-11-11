"""Dynamics/layout checks for the aligned articulated knee device."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase

import mujoco
import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.build_knee_training_model import build_training_model
from tools.preview_knee_hinge_mount import LINKS, build, load_config, update
from myoassist_utils.knee_hinge import joint_name

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_ideal.json"


class KneeTrainingTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = build_training_model(output=Path(cls.temp.name) / "model.xml")
        cls.model = mujoco.MjModel.from_xml_path(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_chain_mass_passive_hinges_and_cuff_weld(self):
        m = self.model
        self.assertEqual((m.nu, m.na, m.nmocap, m.nq), (24, 22, 0, 45))
        mass = 0.
        for side in ('R', 'L'):
            names = [f'hinge_{side}_{link}' for link in LINKS]
            for index, name in enumerate(names):
                body = m.body(name).id
                parent = f'femur_{side.lower()}' if index == 0 else names[index-1]
                self.assertEqual(m.body_parentid[body], m.body(parent).id)
                self.assertEqual(m.body_jntnum[body], int(index > 0))
                self.assertTrue(np.all(m.body_inertia[body] > 0))
                mass += m.body_mass[body]
            self.assertEqual(m.equality(f'knee_cuff_{side.lower()}').type[0], mujoco.mjtEq.mjEQ_WELD)
            for kind in ('upper_hinge', 'lower_hinge'):
                joint = m.joint(joint_name(side, kind)).id
                self.assertNotIn(joint, m.actuator_trnid[m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0])
                np.testing.assert_allclose(m.jnt_range[joint], [-.34907, .34907])
            self.assertEqual(m.actuator(f'Exo_{side}').trnid[0], m.joint(joint_name(side)).id)
        self.assertAlmostEqual(mass, 1.10216)

    def test_geometry_matches_calibrated_preview_at_initial_pose(self):
        pm, pd, devices = build(Path(self.temp.name)/'verify_preview')
        update(pm, pd, devices, load_config(), 0)
        d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, d, 0)
        mujoco.mj_forward(self.model, d)
        for side in ('R','L'):
            for link in LINKS:
                name = f'hinge_{side}_{link}'
                np.testing.assert_allclose(d.xpos[self.model.body(name).id], pd.xpos[pm.body(name).id], atol=1e-8)
                np.testing.assert_allclose(d.xmat[self.model.body(name).id], pd.xmat[pm.body(name).id], atol=1e-8)

    def test_device_and_floor_collisions_are_enabled(self):
        names = ['mount_preview_floor'] + [f'hinge_{side}_{link}_visual_0' for side in ('R','L') for link in LINKS]
        for name in names:
            geom = self.model.geom(name).id
            self.assertEqual(self.model.geom_contype[geom], 1)
            self.assertEqual(self.model.geom_conaffinity[geom], 1)

    def test_neutral_feet_have_ground_contacts(self):
        m = self.model; d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m,d)
        floor = m.geom('mount_preview_floor').id
        self.assertTrue(any(floor in (c.geom1,c.geom2) for c in d.contact))

    def test_motor_torque_cap_and_transmission_to_human_knee(self):
        m = self.model
        for side in ('R','L'):
            human = m.joint(f'knee_angle_{side.lower()}')
            motor_joint = m.joint(joint_name(side))
            actuator = m.actuator(f'Exo_{side}').id
            d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
            base_force=d.qfrc_actuator.copy();base_acc=d.qacc[human.dofadr[0]]
            for command in (-2.,-1.,0.,1.,2.):
                d.ctrl[actuator]=command;mujoco.mj_forward(m,d)
                expected=np.zeros(m.nv);expected[motor_joint.dofadr[0]]=20*np.clip(command,-1,1)
                np.testing.assert_allclose(d.qfrc_actuator-base_force,expected,atol=1e-8)
                self.assertAlmostEqual(d.actuator_force[actuator],20*np.clip(command,-1,1))
                if command:
                    self.assertLess((d.qacc[human.dofadr[0]]-base_acc)*command,0)

    def test_keyframes_map_human_and_device_joints_by_name(self):
        m=self.model
        for key in range(m.nkey):
            for side in ('r','l'):
                device=m.joint(joint_name(side)).qposadr[0];human=m.joint('knee_angle_'+side).qposadr[0]
                self.assertAlmostEqual(m.key_qpos[key,device],-m.key_qpos[key,human])

    def test_config_layout_and_mirror_map_for_custom_xml(self):
        c=DictionableDataclass.create(ExoImitationTrainSessionConfig,json.loads(CONFIG.read_text()))
        c.env_params.model_path=str(self.path)
        EnvironmentHandler._validate_action_layout(str(self.path),c)
        obs,action,muscles=EnvironmentHandler._mirror_permutations(c)
        self.assertEqual((len(obs),len(action),muscles),(44,24,22))
        self.assertEqual(list(action[22:]),[23,22])
