import glob
import os
import shutil
import sys
import unittest
import dpdata
import numpy as np
from monty.serialization import loadfn
from apex.core.property.Lat_param_T import Lat_param_T
from apex.core.calculator.Lammps import Lammps
from apex.core.calculator.lib.lammps_utils import inter_deepmd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
__package__ = "tests"


class Test_Lat_param_T(unittest.TestCase):
    def setUp(self):
        _jdata = {
            "structures": ["confs/hcp-Ti"],
            "interaction": {
                "type": "meam_spline",
                "model": "lammps_input/Ti.meam.spline",
                "type_map": {"Ti": 0}
            },
            "properties": [
                {
                    "type": "Lat_param_T",
                    "supercell_size":  [2, 2, 2],
                    "cal_setting":{
                        "temperature": [400, 600],
                        "equi_step": 4000,
                        "N_every": 100,
                        "N_repeat": 5,
                        "N_freq": 1000,
                        "ave_step": 4000
                    }
                }
            ],
        }

        self.equi_path = "confs/hcp-Ti/relaxation/relax_task"
        self.source_path = "equi/lammps"
        self.target_path = "confs/hcp-Ti/Lat_param_T_00"

        if not os.path.exists(self.equi_path):
            os.makedirs(self.equi_path)

        self.confs = _jdata["structures"]
        self.inter_param = _jdata["interaction"]
        self.prop_param = _jdata["properties"]

        self.Lat_param_T = Lat_param_T(_jdata["properties"][0])
        self.Lammps = Lammps(
            self.inter_param, os.path.join(self.source_path, "hcp-Ti-CONTCAR")
        )

    def tearDown(self):
        if os.path.exists(os.path.abspath(os.path.join(self.equi_path, ".."))):
            shutil.rmtree(os.path.abspath(os.path.join(self.equi_path, "..")))
        if os.path.exists(self.equi_path):
            shutil.rmtree(self.equi_path)
        if os.path.exists(self.target_path):
            shutil.rmtree(self.target_path)

    def test_task_type(self):
        self.assertEqual("Lat_param_T", self.Lat_param_T.task_type())

    def test_task_param(self):
        self.assertEqual(self.prop_param[0], self.Lat_param_T.task_param())

    def test_make_potential_files(self):
        cwd = os.getcwd()
        abs_equi_path = os.path.abspath(self.equi_path)
        self.Lammps.make_potential_files(abs_equi_path)
        self.assertTrue(os.path.islink(os.path.join(self.equi_path, "Ti.meam.spline")))
        self.assertTrue(os.path.isfile(os.path.join(self.equi_path, "inter.json")))
        ret = loadfn(os.path.join(self.equi_path, "inter.json"))
        self.assertEqual(self.inter_param, ret)
        os.chdir(cwd)

    def test_make_confs(self):
        if not os.path.exists(os.path.join(self.equi_path, "CONTCAR")):
            with self.assertRaises(RuntimeError):
                self.Lat_param_T.make_confs(self.target_path, self.equi_path)
        shutil.copy(
            os.path.join(self.source_path, "hcp-Ti-CONTCAR"),
            os.path.join(self.equi_path, "CONTCAR"),
        )

        task_list = self.Lat_param_T.make_confs(self.target_path, self.equi_path)
        self.assertEqual(len(task_list), 2)
        dfm_dirs = glob.glob(os.path.join(self.target_path, "task.*"))
        num = 0
        dfm_dirs.sort()

        for ii in dfm_dirs:
            self.assertTrue(os.path.isfile(os.path.join(ii, "POSCAR")))
            Lat_param_T_json_file = os.path.join(ii, "Lat_param_T.json")
            self.assertTrue(os.path.isfile(Lat_param_T_json_file))
            variable_Lat_param_T_file = os.path.join(ii, "variable_Lat_param_T.in")
            self.assertTrue(os.path.isfile(variable_Lat_param_T_file))
            with open(variable_Lat_param_T_file, 'r') as file:
                lines = file.readlines()
                temp = lines[1].strip()
            self.assertEqual(temp, "variable temperature equal %.2f" % self.prop_param[0]["cal_setting"]["temperature"][num])
            num += 1

    def test_forward_common_files(self):
        fc_files = ["in.lammps", "variable_Lat_param_T.in", "Ti.meam.spline"]
        self.assertEqual(self.Lammps.forward_common_files(self.prop_param[0]["type"]), fc_files)

    def test_backward_files(self):
        backward_files = ["log.lammps", "outlog", "dump.relax", "average_box.txt"]
        self.assertEqual(self.Lammps.backward_files(self.prop_param[0]["type"]), backward_files)

