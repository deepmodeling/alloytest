import os
import warnings
import logging

from monty.serialization import dumpfn, loadfn
from contextlib import contextmanager
from apex.core.calculator.lib import lammps_utils
from apex.core.calculator.lib.lammps_utils import (
    inter_deepmd,
    inter_eam_alloy,
    inter_eam_fs,
    inter_meam,
    inter_meam_spline,
    inter_snap,
    inter_gap,
    inter_rann,
    inter_mace,
    inter_nep
)
from .Task import Task
from dflow.python import upload_packages
from . import LAMMPS_INTER_TYPE
upload_packages.append(__file__)

# LAMMPS_INTER_TYPE = ['deepmd', 'eam_alloy', 'meam', 'eam_fs', 'meam_spline', 'snap', 'gap', 'rann', 'mace']
MULTI_MODELS_INTER_TYPE = ["meam", "snap", "gap"]

class Lammps(Task):
    def __init__(self, inter_parameter, path_to_poscar):
        self.inter = inter_parameter
        self.inter_type = inter_parameter["type"]
        self.type_map = inter_parameter["type_map"]
        self.in_lammps = inter_parameter.get("in_lammps", "auto")
        if self.inter_type in MULTI_MODELS_INTER_TYPE:
            self.model = list(map(os.path.abspath, inter_parameter["model"]))
        else:
            self.model = os.path.abspath(inter_parameter["model"])
        self.path_to_poscar = path_to_poscar
        assert self.inter_type in LAMMPS_INTER_TYPE
        self.set_inter_type_func()

    def set_inter_type_func(self):
        if self.inter_type == "deepmd":
            self.inter_func = inter_deepmd
        elif self.inter_type == "eam_fs":
            self.inter_func = inter_eam_fs
        elif self.inter_type == "meam":
            self.inter_func = inter_meam
        elif self.inter_type == "meam_spline":
            self.inter_func = inter_meam_spline
        elif self.inter_type == "snap":
            self.inter_func = inter_snap
        elif self.inter_type == "gap":
            self.inter_func = inter_gap
        elif self.inter_type == "rann":
            self.inter_func = inter_rann
        elif self.inter_type == "mace":
            self.inter_func = inter_mace
        elif self.inter_type == "nep":
            self.inter_func = inter_nep
        else:
            self.inter_func = inter_eam_alloy

    def set_model_param(self):
        deepmd_version = self.inter.get("deepmd_version", "2.1.1")
        if self.inter_type == "deepmd":
            model_name = os.path.basename(self.model)
            self.model_param = {
                "type": self.inter_type,
                "model_name": [model_name],
                "param_type": self.type_map,
                "deepmd_version": deepmd_version,
            }
        elif self.inter_type in ["meam", "snap"]:
            model_name = list(map(os.path.basename, self.model))
            self.model_param = {
                "type": self.inter_type,
                "model_name": model_name,
                "param_type": self.type_map,
                "deepmd_version": deepmd_version,
            }
        elif self.inter_type == "gap":
            model_name = list(map(os.path.basename, self.model))
            self.model_param = {
                "type": self.inter_type,
                "model_name": model_name,
                "param_type": self.type_map,
                "init_string": self.inter.get("init_string", None),
                "atomic_num_list": self.inter.get("atomic_num_list", None),
                "deepmd_version": deepmd_version,
            }
        else:
            model_name = os.path.basename(self.model)
            self.model_param = {
                "type": self.inter_type,
                "model_name": [model_name],
                "param_type": self.type_map,
                "deepmd_version": deepmd_version,
            }

    @contextmanager
    def change_dir(self,path):
        old_dir = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old_dir)

    def symlink_force(self, target, link_name):
        if os.path.islink(link_name) or os.path.exists(link_name):
            os.remove(link_name)
        os.symlink(target, link_name)

    def make_potential_files(self, output_dir):
        parent_dir = os.path.join(output_dir, "../../")
        if self.inter_type in MULTI_MODELS_INTER_TYPE:
            model_file = map(os.path.basename, self.model)
            targets = self.model
            link_names = list(model_file)
        else:
            model_file = os.path.basename(self.model)
            targets = [self.model]
            link_names = [model_file]

        with self.change_dir(parent_dir):
            for target, link_name in zip(targets, link_names):
                self.symlink_force(os.path.relpath(target), link_name)

        with self.change_dir(output_dir):
            for link_name in link_names:
                self.symlink_force(os.path.join("../../", link_name), link_name)

        dumpfn(self.inter, os.path.join(output_dir, "inter.json"), indent = 4)

    def make_input_file(self, output_dir, task_type, task_param):
        lammps_utils.cvt_lammps_conf(
            os.path.join(output_dir, "POSCAR"),
            os.path.join(output_dir, "conf.lmp"),
            lammps_utils.element_list(self.type_map),
        )

        cal_type = task_param["cal_type"]
        cal_setting = task_param["cal_setting"]
        prop_type = task_param.get("type", "relaxation")

        etol = cal_setting.get("etol", 0)
        ftol = cal_setting.get("ftol", 1e-10)
        maxiter = cal_setting.get("maxiter", 5000)
        maxeval = cal_setting.get("maxeval", 500000)
        B0 = 70
        bp = 0

        self.set_model_param()

        # deal with user input in.lammps for relaxation
        if os.path.isfile(self.in_lammps) and task_type == "relaxation":
            with open(self.in_lammps, "r") as fin:
                fc = fin.read()
        # user input in.lammps for apex calculation
        elif "input_prop" in cal_setting \
                and os.path.isfile(cal_setting["input_prop"]) \
                and not task_type == "relaxation":
            with open(os.path.abspath(cal_setting["input_prop"]), "r") as fin:
                fc = fin.read()

        else:
            logging.basicConfig(level=logging.INFO, filename='info.log', filemode='w',
                                format='%(asctime)s - %(levelname)s - %(message)s')
            if "etol" in cal_setting:
                logging.info(
                    "%s setting etol to %s"
                    % (self.make_input_file.__name__, cal_setting["etol"])
                )
                etol = cal_setting["etol"]
            if "ftol" in cal_setting:
                logging.info(
                    "%s setting ftol to %s"
                    % (self.make_input_file.__name__, cal_setting["ftol"])
                )
                ftol = cal_setting["ftol"]
            if "maxiter" in cal_setting:
                logging.info(
                    "%s setting maxiter to %s"
                    % (self.make_input_file.__name__, cal_setting["maxiter"])
                )
                maxiter = cal_setting["maxiter"]
            if "maxeval" in cal_setting:
                logging.info(
                    "%s setting maxeval to %s"
                    % (self.make_input_file.__name__, cal_setting["maxeval"])
                )
                maxeval = cal_setting["maxeval"]

            if cal_type == "relaxation":
                relax_pos = cal_setting["relax_pos"]
                relax_shape = cal_setting["relax_shape"]
                relax_vol = cal_setting["relax_vol"]

                if [relax_pos, relax_shape, relax_vol] == [True, False, False]:
                    fc = lammps_utils.make_lammps_equi(
                        "conf.lmp",
                        self.type_map,
                        self.inter_func,
                        self.model_param,
                        etol,
                        ftol,
                        maxiter,
                        maxeval,
                        False,
                        prop_type=prop_type,
                    )
                elif [relax_pos, relax_shape, relax_vol] == [True, True, True]:
                    fc = lammps_utils.make_lammps_equi(
                        "conf.lmp",
                        self.type_map,
                        self.inter_func,
                        self.model_param,
                        etol,
                        ftol,
                        maxiter,
                        maxeval,
                        True,
                        prop_type=prop_type,
                    )
                elif [relax_pos, relax_shape, relax_vol] == [
                    True,
                    True,
                    False,
                ] and not task_type == "eos":
                    if "scale2equi" in task_param:
                        scale2equi = task_param["scale2equi"]
                        fc = lammps_utils.make_lammps_press_relax(
                            "conf.lmp",
                            self.type_map,
                            scale2equi[int(output_dir[-6:])],
                            self.inter_func,
                            self.model_param,
                            B0,
                            bp,
                            etol,
                            ftol,
                            maxiter,
                            maxeval,
                        )
                    else:
                        fc = lammps_utils.make_lammps_equi(
                            "conf.lmp",
                            self.type_map,
                            self.inter_func,
                            self.model_param,
                            etol,
                            ftol,
                            maxiter,
                            maxeval,
                            True,
                        )
                elif [relax_pos, relax_shape, relax_vol] == [
                    True,
                    True,
                    False,
                ] and task_type == "eos":
                    task_param["cal_setting"]["relax_shape"] = False
                    fc = lammps_utils.make_lammps_equi(
                        "conf.lmp",
                        self.type_map,
                        self.inter_func,
                        self.model_param,
                        etol,
                        ftol,
                        maxiter,
                        maxeval,
                        False,
                        prop_type=prop_type,
                    )
                elif [relax_pos, relax_shape, relax_vol] == [False, False, False]:
                    fc = lammps_utils.make_lammps_eval(
                        "conf.lmp", self.type_map, self.inter_func, self.model_param
                    )

                else:
                    raise RuntimeError("not supported calculation setting for LAMMPS")

            elif cal_type == "static":
                fc = lammps_utils.make_lammps_eval(
                    "conf.lmp", self.type_map, self.inter_func, self.model_param
                )

            else:
                raise RuntimeError("not supported calculation type for LAMMPS")

        dumpfn(task_param, os.path.join(output_dir, "task.json"), indent=4)

        in_lammps_not_link_list = ["eos"]
        if task_type not in in_lammps_not_link_list:
            with open(os.path.join(output_dir, "../in.lammps"), "w") as fp:
                fp.write(fc)
            cwd = os.getcwd()
            os.chdir(output_dir)
            if not (os.path.islink("in.lammps") or os.path.isfile("in.lammps")):
                os.symlink("../in.lammps", "in.lammps")
            else:
                os.remove("in.lammps")
                os.symlink("../in.lammps", "in.lammps")
            os.chdir(cwd)
        else:
            with open(os.path.join(output_dir, "in.lammps"), "w") as fp:
                fp.write(fc)

    def compute(self, output_dir):
        log_lammps = os.path.join(output_dir, "log.lammps")
        dump_lammps = os.path.join(output_dir, "dump.relax")
        if not os.path.isfile(log_lammps) or not os.path.isfile(dump_lammps):
            warnings.warn(f"cannot find {'log.lammps' if not os.path.isfile(log_lammps) else 'dump.relax'} in {output_dir} skip")
            return None

        box, coord, vol, energy, force, virial, stress = [], [], [], [], [], [], []
        dumptime, type_list = self._parse_dump_file(dump_lammps, box, coord, vol, force)
        
        if not self._check_lammps_finished(log_lammps):
            return None
        
        self._parse_log_file(log_lammps, dumptime, energy, stress, virial, vol)

        type_map_list = lammps_utils.element_list(self.type_map)
        atom_numbs = self._calculate_atom_numbers(type_list, len(type_map_list))

        result_dict = self._prepare_result_dict(atom_numbs, type_map_list, type_list, box, coord, energy, force, virial, stress)
        contcar = os.path.join(output_dir, "CONTCAR")
        dumpfn(result_dict, contcar, indent=4)
        d_dump = loadfn(contcar)
        d_dump.to("vasp/poscar", contcar, frame_idx=-1)

        return result_dict
    
    def _parse_dump_file(self, dump_lammps, box, coord, vol, force):
        with open(dump_lammps, "r") as fin:
            dump = fin.read().split("\n")
        dumptime = []
        for idx, ii in enumerate(dump):
            if ii == "ITEM: TIMESTEP":
                box.append([])
                coord.append([])
                force.append([])
                dumptime.append(int(dump[idx + 1]))
                natom = int(dump[idx + 3])
                xlo_bound = float(dump[idx + 5].split()[0])
                xhi_bound = float(dump[idx + 5].split()[1])
                xy = float(dump[idx + 5].split()[2])
                ylo_bound = float(dump[idx + 6].split()[0])
                yhi_bound = float(dump[idx + 6].split()[1])
                xz = float(dump[idx + 6].split()[2])
                zlo = float(dump[idx + 7].split()[0])
                zhi = float(dump[idx + 7].split()[1])
                yz = float(dump[idx + 7].split()[2])
                xx = (
                    xhi_bound
                    - max([0, xy, xz, xy + xz])
                    - (xlo_bound - min([0, xy, xz, xy + xz]))
                )
                yy = yhi_bound - max([0, yz]) - (ylo_bound - min([0, yz]))
                zz = zhi - zlo
                box[-1].append([xx, 0.0, 0.0])
                box[-1].append([xy, yy, 0.0])
                box[-1].append([xz, yz, zz])
                vol.append(xx * yy * zz)
                type_list = []
                for jj in range(natom):
                    type_list.append(int(dump[idx + 9 + jj].split()[1]) - 1)
                    if "xs ys zs" in dump[idx + 8]:
                        a_x = (
                            float(dump[idx + 9 + jj].split()[2]) * xx
                            + float(dump[idx + 9 + jj].split()[3]) * xy
                            + float(dump[idx + 9 + jj].split()[4]) * xz
                        )
                        a_y = (
                            float(dump[idx + 9 + jj].split()[3]) * yy
                            + float(dump[idx + 9 + jj].split()[4]) * yz
                        )
                        a_z = float(dump[idx + 9 + jj].split()[4]) * zz
                    else:
                        a_x = float(dump[idx + 9 + jj].split()[2])
                        a_y = float(dump[idx + 9 + jj].split()[3])
                        a_z = float(dump[idx + 9 + jj].split()[4])
                    coord[-1].append([a_x, a_y, a_z])
                    fx = float(dump[idx + 9 + jj].split()[5])
                    fy = float(dump[idx + 9 + jj].split()[6])
                    fz = float(dump[idx + 9 + jj].split()[7])
                    force[-1].append([fx, fy, fz])
        
        return dumptime, type_list
    
    def _check_lammps_finished(self, log_lammps):
        with open(log_lammps, "r") as fp:
            if "Total wall time:" not in fp.read():
                warnings.warn("lammps not finished " + log_lammps + " skip")
                return False
        return True

    def _parse_log_file(self, log_lammps, dumptime, energy, stress, virial, vol):
        with open(log_lammps, "r") as fp:
            fp.seek(0)
            lines = fp.read().split("\n")
            idid = -1
            for ii in dumptime:
                idid += 1
                for jj in lines:
                    line = jj.split()
                    if len(line) and str(ii) == line[0]:
                        try:
                            [float(kk) for kk in line]
                        except Exception:
                            continue
                        stress.append([])
                        virial.append([])
                        energy.append(float(line[1]))
                        # virials = stress * vol * 1e5 *1e-30 * 1e19/1.6021766208
                        stress[-1].append(
                            [
                                float(line[2]) / 1000.0,
                                float(line[5]) / 1000.0,
                                float(line[6]) / 1000.0,
                            ]
                        )
                        stress[-1].append(
                            [
                                float(line[5]) / 1000.0,
                                float(line[3]) / 1000.0,
                                float(line[7]) / 1000.0,
                            ]
                        )
                        stress[-1].append(
                            [
                                float(line[6]) / 1000.0,
                                float(line[7]) / 1000.0,
                                float(line[4]) / 1000.0,
                            ]
                        )
                        stress_to_virial = (
                            vol[idid] * 1e5 * 1e-30 * 1e19 / 1.6021766208
                        )
                        virial[-1].append(
                            [
                                float(line[2]) * stress_to_virial,
                                float(line[5]) * stress_to_virial,
                                float(line[6]) * stress_to_virial,
                            ]
                        )
                        virial[-1].append(
                            [
                                float(line[5]) * stress_to_virial,
                                float(line[3]) * stress_to_virial,
                                float(line[7]) * stress_to_virial,
                            ]
                        )
                        virial[-1].append(
                            [
                                float(line[6]) * stress_to_virial,
                                float(line[7]) * stress_to_virial,
                                float(line[4]) * stress_to_virial,
                            ]
                        )
                        break

    def _calculate_atom_numbers(self, type_list, type_map_length):
        atom_numbs = [0] * type_map_length
        for atom_type in type_list:
            atom_numbs[atom_type] += 1
        return atom_numbs
    
    def _prepare_result_dict(self, atom_numbs, type_map_list, type_list, box, coord, energy, force, virial, stress):

        result_dict = {
            "@module": "dpdata.system",
            "@class": "LabeledSystem",
            "data": {
                "atom_numbs": atom_numbs,
                "atom_names": type_map_list,
                "atom_types": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "int64",
                    "data": type_list,
                },
                "orig": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "int64",
                    "data": [0, 0, 0],
                },
                "cells": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": box,
                },
                "coords": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": coord,
                },
                "energies": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": energy,
                },
                "forces": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": force,
                },
                "virials": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": virial,
                },
                "stress": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": stress,
                },
            },
        }
        return result_dict

    def forward_files(self, property_type="relaxation"):
        if self.inter_type in MULTI_MODELS_INTER_TYPE:
            return ["conf.lmp", "in.lammps"] + list(map(os.path.basename, self.model))
        elif property_type == "Lat_param_T":
            return ["in.lammps", "variable_Lat_param_T.in", os.path.basename(self.model)]
        else:
            return ["conf.lmp", "in.lammps", os.path.basename(self.model)]

    def forward_common_files(self, property_type="relaxation"):
        if property_type not in ["eos"]:
            if self.inter_type in MULTI_MODELS_INTER_TYPE:
                return ["in.lammps"] + list(map(os.path.basename, self.model))
            elif property_type == "Lat_param_T":
                return ["in.lammps", "variable_Lat_param_T.in", os.path.basename(self.model)]
            else:
                return ["in.lammps", os.path.basename(self.model)]
        else:
            if self.inter_type in MULTI_MODELS_INTER_TYPE:
                return list(map(os.path.basename, self.model))
            else:
                return [os.path.basename(self.model)]

    def backward_files(self, property_type="relaxation"):
        if property_type == "phonon":
            return ["outlog", "FORCE_CONSTANTS"]
        elif property_type == "Lat_param_T":
            return ["log.lammps", "outlog", "dump.relax", "average_box.txt"]
        else:
            return ["log.lammps", "outlog", "dump.relax"]

