import glob
import json
import logging
import os
import re
import dpdata
import numpy as np
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from apex.core.calculator.lib import abacus_utils
from apex.core.calculator.lib import vasp_utils
from apex.core.property.Property import Property
from apex.core.refine import make_refine
from apex.core.reproduce import make_repro, post_repro
from dflow.python import upload_packages
from monty.serialization import loadfn, dumpfn
upload_packages.append(__file__)

class Lat_param_T(Property):
    def __init__(self, parameter, inter_param=None):
        '''
        Lammps
        supercell_size = [2,2,2]
        temperature = [700, 800, 900, 1000]
        equi_step = 80000
        N_every = 100
        N_repeat = 10
        N_freq = 2000
        ave_step = 40000
        "cal_type" = "npt+ave/time"
        '''
        parameter["reproduce"] = parameter.get("reproduce", False)
        self.reprod = parameter["reproduce"]
        # default_cal_setting_input_prop = {"input_prop": "lammps_input/lat_param_T/in.lammps"}
        if "cal_setting" not in parameter:
            parameter["cal_setting"] = {}
            # parameter["cal_setting"]["input_prop"] = "lammps_input/lat_param_T/in.lammps"
        if not self.reprod:
            if not ("init_from_suffix" in parameter and "output_suffix" in parameter):
                parameter["supercell_size"] = parameter.get("supercell_size",[2,2,2])
                self.supercell_size = parameter["supercell_size"]
                parameter["cal_setting"]["temperature"] = parameter["cal_setting"].get("temperature", [200,400,600,800])
                parameter["cal_setting"]["equi_step"] = parameter["cal_setting"].get("equi_step", 80000)
                parameter["cal_setting"]["N_every"] = parameter["cal_setting"].get("N_every", 100)
                parameter["cal_setting"]["N_repeat"] = parameter["cal_setting"].get("N_repeat", 10)
                parameter["cal_setting"]["N_freq"] = parameter["cal_setting"].get("N_freq", 2000)
                parameter["cal_setting"]["ave_step"] = parameter["cal_setting"].get("ave_step", 40000)
        else:
            parameter["init_from_suffix"] = parameter.get("init_from_suffix", "00")
            self.init_from_suffix = parameter["init_from_suffix"]
        self.cal_setting = parameter["cal_setting"]
        parameter["cal_type"] = "npt+ave/time"
        self.parameter = parameter
        self.inter_param = inter_param if inter_param != None else {"type": "lammps"}

    def make_confs(self,path_to_work, path_to_equi, refine=False):
        path_to_work = os.path.abspath(path_to_work)
        if os.path.exists(path_to_work):
            logging.warning('%s already exists' % path_to_work)
        else:
            os.makedirs(path_to_work)
        path_to_equi = os.path.abspath(path_to_equi)

        task_list = []
        cwd = os.getcwd()

        if self.reprod:
            print("surface reproduce starts")
            if "init_data_path" not in self.parameter:
                raise RuntimeError("please provide the initial data path to reproduce")
            init_data_path = os.path.abspath(self.parameter["init_data_path"])
            task_list = make_repro(
                self.inter_param,
                init_data_path,
                self.init_from_suffix,
                path_to_work,
                self.parameter.get("reprod_last_frame", True),
            )
        else:
            if refine:
                logging.info("Lat_param_T refine starts")
                task_list = make_refine(
                    self.parameter["init_from_suffix"],
                    self.parameter["output_suffix"],
                    path_to_work,
                )
                init_from_path = re.sub(
                    self.parameter["output_suffix"][::-1],
                    self.parameter["init_from_suffix"][::-1],
                    path_to_work[::-1],
                    count=1,
                )[::-1]
                task_list_basename = list(map(os.path.basename, task_list))

                for ii in task_list_basename:
                    init_from_task = os.path.join(init_from_path, ii)
                    output_task = os.path.join(path_to_work, ii)
                    os.chdir(output_task)
                    if os.path.isfile("variable_Lat_param_T.json"):
                        os.remove("variable_Lat_param_T.json")
                    if os.path.islink("variable_Lat_param_T.json"):
                        os.remove("variable_Lat_param_T.json")
                    os.symlink(
                        os.path.relpath(os.path.join(init_from_task, "variable_Lat_param_T.json")),
                        "variable_Lat_param_T.json",)
            else:
                if self.inter_param["type"] == "abacus":
                    raise TypeError("Lat_param_T only support lammps calculation")
                else:
                # refine = false && reproduce = false && self.inter_param["type"]== "vasp"
                    CONTCAR = "CONTCAR"
                    POSCAR = "POSCAR"
                equi_contcar = os.path.join(path_to_equi, CONTCAR)
                if not os.path.exists(equi_contcar):
                    raise RuntimeError("please do relaxation first")
                # get structure
                if self.inter_param["type"] == "abacus":
                    raise TypeError("Lat_param_T only support lammps calculation")
                else:
                    ptypes = vasp_utils.get_poscar_types(equi_contcar)
                    ss = Structure.from_file(equi_contcar)
                '''
                based on temperature build the dir
                copy the contcar in equi to each dir              
                build variable_Lat_param_T.json
                '''
                for ii in range(len(self.parameter["cal_setting"]["temperature"])):
                    output_task = os.path.join(path_to_work, "task.%06d" % ii)
                    os.makedirs(output_task, exist_ok=True)
                    os.chdir(output_task)
                    for jj in [
                        "INCAR",
                        "POTCAR",
                        "POSCAR",
                        "conf.lmp",
                        "in.lammps",
                        "STRU",
                    ]:
                        if os.path.exists(jj):
                            os.remove(jj)
                    task_list.append(output_task)
                    # build the structure for every task
                    # initially I want to directly copy CONTCAR from relaxation as shutil package is not included, another
                    # method is used here
                    ss.to("POSCAR.tmp", "POSCAR")
                    vasp_utils.regulate_poscar("POSCAR.tmp", "POSCAR")
                    vasp_utils.sort_poscar("POSCAR", "POSCAR", ptypes)
                    # Lat_param_T.json
                    temp = self.parameter["cal_setting"]["temperature"][ii]
                    Lat_param_T_task = {"temperature":temp, "supercell_size":self.supercell_size}
                    dumpfn(Lat_param_T_task, "Lat_param_T.json", indent=4)
                    # variable_Lat_param_T.txt
                    ret = self._variable(temp)
                    with open("variable_Lat_param_T.in", "w") as fp:
                        fp.write(ret)
        os.chdir(cwd)
        return task_list

    def post_process(self, task_list):
        pass

    def task_type(self):
        return self.parameter['type']

    def task_param(self):
        return self.parameter

    def _compute_lower(self,
                       output_file,
                       all_tasks,
                       all_res):
        output_file = os.path.abspath(output_file)
        res_data = {}
        ptr_data = os.path.dirname(output_file) + "\n"

        if not self.reprod:
            ptr_data += ' Temperature(K)  a(A)  b(A)  c(A)  c/a\n'
            num = 0
            for ii in all_tasks:
                a_sum = 0
                b_sum = 0
                c_sum = 0
                count = 0
                temp = self.parameter["cal_setting"]["temperature"][num]
                supercell_size = self.supercell_size
                num += 1
                with open(os.path.join(ii, "average_box.txt"), 'r') as file:
                    for line in file:
                        if line.startswith("#") or line.strip() == "":
                            continue
                        parts = line.split()
                        if len(parts) == 4:
                            timestep, v_lx, v_ly, v_lz = parts
                            a_sum += float(v_lx)
                            b_sum += float(v_ly)
                            c_sum += float(v_lz)
                            count += 1
                a = a_sum / count if count != 0 else 0
                b = b_sum / count if count != 0 else 0
                c = c_sum / count if count != 0 else 0

                a = a / supercell_size[0]
                b = b / supercell_size[1]
                c = c / supercell_size[2]

                structure_dir = os.path.basename(ii)

                ptr_data += "%-25s  %7.6f  %7.6f  %7.6f \n" %(
                    str(temp) + ":",
                    a, b, c
                )

                res_data[str(temp)] = [
                    a, b, c, temp
                ]

            with open(output_file, 'w') as fp:
                json.dump(res_data, fp, indent=4)
        else:
            if "init_data_path" not in self.parameter:
                raise RuntimeError("please provide the initial data path to reproduce")
            init_data_path = os.path.abspath(self.parameter["init_data_path"])
            res_data, ptr_data = post_repro(
                init_data_path,
                self.parameter["init_from_suffix"],
                all_tasks,
                ptr_data,
                self.parameter.get("reprod_last_frame", True),
            )

        with open(output_file, "w") as fp:
            json.dump(res_data, fp, indent=4)

        return res_data, ptr_data

    def _variable(self, temp):
        ret = ""
        ret += " # variable_Lat_param_T.in \n"
        ret += "variable temperature equal %.2f\n" %temp
        ret += "variable nx equal %d\n" % self.supercell_size[0]
        ret += "variable ny equal %d\n" % self.supercell_size[1]
        ret += "variable nz equal %d\n" % self.supercell_size[2]
        ret += "variable equi_step equal %d\n" % self.parameter["cal_setting"]["equi_step"]
        ret += "variable N_every equal %d\n" % self.parameter["cal_setting"]["N_every"]
        ret += "variable N_repeat equal %d\n" % self.parameter["cal_setting"]["N_repeat"]
        ret += "variable N_freq equal %d\n" % self.parameter["cal_setting"]["N_freq"]
        ret += "variable ave_step equal %d\n" % self.parameter["cal_setting"]["ave_step"]
        return ret

