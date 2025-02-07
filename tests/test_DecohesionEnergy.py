import glob
import os
import shutil
import sys
import unittest

from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Incar

from apex.core.property.DecohesionEnergy import DecohesionEnergy
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
__package__ = "tests"


class TestDecohesionEnergy(unittest.TestCase):
    def setUp(self):
        _jdata = {
            "structures": ["confs/mp-141"],
            "interaction": {
                "type": "vasp",
                "incar": "vasp_input/INCAR.rlx",
                "potcar_prefix": ".",
                "potcars": {"Yb": "vasp_input/POTCAR"},
            },
            "properties": [
                {
                    "type": "DecohesionEnergy",
                    "min_slab_size": 15,
                    "max_vacuum_size": 10,
                    "vacuum_size_step": 5,
                    "miller_index":   [0, 0, 1],
                    "cal_type":       "static"
                }
            ],
        }

        self.equi_path = "confs/mp-141/relaxation/relax_task"
        self.source_path = "equi/vasp"
        self.target_path = "confs/mp-141/DecohesionEnergy_00"
        if not os.path.exists(self.equi_path):
            os.makedirs(self.equi_path)

        self.confs = _jdata["structures"]
        self.inter_param = _jdata["interaction"]
        self.prop_param = _jdata["properties"]

        self.DecohesionEnergy = DecohesionEnergy(_jdata["properties"][0])

    def tearDown(self):
        if os.path.exists(os.path.abspath(os.path.join(self.equi_path, ".."))):
            shutil.rmtree(os.path.abspath(os.path.join(self.equi_path, "..")))
        if os.path.exists(self.equi_path):
            shutil.rmtree(self.equi_path)
        if os.path.exists(self.target_path):
            shutil.rmtree(self.target_path)

    def test_task_type(self):
        self.assertEqual("DecohesionEnergy", self.DecohesionEnergy.task_type())

    def test_task_param(self):
        self.assertEqual(self.prop_param[0], self.DecohesionEnergy.task_param())

    def test_make_confs_0(self):
        if not os.path.exists(os.path.join(self.equi_path, "CONTCAR")):
            with self.assertRaises(RuntimeError):
                self.DecohesionEnergy.make_confs(self.target_path, self.equi_path)
        shutil.copy(
            os.path.join(self.source_path, "mp-141.vasp"),
            os.path.join(self.equi_path, "CONTCAR"),
        )
        task_list = self.DecohesionEnergy.make_confs(self.target_path, self.equi_path)
        self.assertEqual(len(task_list), 3)
        dfm_dirs = glob.glob(os.path.join(self.target_path, "task.*"))

        incar0 = Incar.from_file(os.path.join("vasp_input", "INCAR.rlx"))
        incar0["ISIF"] = 2
        incar0["NSW"] = 0

        self.assertEqual(
            os.path.realpath(os.path.join(self.equi_path, "CONTCAR")),
            os.path.realpath(os.path.join(self.target_path, "POSCAR")),
        )
        ref_st = Structure.from_file(os.path.join(self.target_path, "POSCAR"))
        dfm_dirs.sort()
        num = 0
        for ii in dfm_dirs:
            st_file = os.path.join(ii, "POSCAR")
            self.assertTrue(os.path.isfile(st_file))
            st1_file = os.path.join(ii, "POSCAR.tmp")
            self.assertTrue(os.path.isfile(st1_file))
            st1 = Structure.from_file(st1_file)
            decohesion_energy_json_file = os.path.join(ii, "decohesion_energy.json")
            decohesion_energy_json = loadfn(decohesion_energy_json_file)
            sl = self.__gen_slab_pmg(
                ref_st,
                tuple(decohesion_energy_json["miller_index"]),
                self.prop_param[0]["min_slab_size"],
                self.prop_param[0]["vacuum_size_step"] * num,
            )
            num += 1
            # slb = sl.get_slab()
            st2 = Structure(sl.lattice, sl.species, sl.frac_coords)
            self.assertEqual(len(st1), len(st2))

    def __gen_slab_pmg(self, structure: Structure,
                       plane_miller, slab_size, vacuum_size) -> Structure:

        # Generate slab via Pymatgen
        slabGen = SlabGenerator(structure, miller_index=plane_miller,
                                min_slab_size=slab_size, min_vacuum_size=0,
                                center_slab=True, in_unit_planes=False,
                                lll_reduce=True, reorient_lattice=False,
                                primitive=False)
        slabs_pmg = slabGen.get_slabs(ftol=0.001)
        slab = [s for s in slabs_pmg if s.miller_index == plane_miller][0]
        # If a transform matrix is passed, reorient the slab
        order = zip(slab.frac_coords, slab.species)
        c_order = sorted(order, key=lambda x: x[0][2])
        sorted_frac_coords = []
        sorted_species = []
        for (frac_coord, species) in c_order:
            sorted_frac_coords.append(frac_coord)
            sorted_species.append(species)
        # add vacuum layer to the slab with height unit of angstrom
        a, b, c = slab.lattice.matrix
        slab_height = slab.lattice.matrix[2][2]
        if slab_height >= 0:
            self.is_flip = False
            elong_scale = 1 + (vacuum_size / slab_height)
        else:
            self.is_flip = True
            elong_scale = 1 + (-vacuum_size / slab_height)
        new_lattice = [a, b, elong_scale * c]
        new_frac_coords = []
        for ii in range(len(sorted_frac_coords)):
            coord = sorted_frac_coords[ii].copy()
            coord[2] = coord[2] / elong_scale
            new_frac_coords.append(coord)
        slab_new = Structure(lattice=np.array(new_lattice),
                         coords=new_frac_coords, species=sorted_species)

        return slab_new

