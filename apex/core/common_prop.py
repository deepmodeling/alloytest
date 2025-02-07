import glob
import os
from multiprocessing import Pool
from monty.serialization import dumpfn, loadfn

from apex.core.calculator.calculator import make_calculator
from apex.core.property.Elastic import Elastic
from apex.core.property.EOS import EOS
from apex.core.property.Gamma import Gamma
from apex.core.property.Interstitial import Interstitial
from apex.core.property.Surface import Surface
from apex.core.property.Vacancy import Vacancy
from apex.core.property.Phonon import Phonon
from apex.core.property.DecohesionEnergy import DecohesionEnergy
from apex.core.property.Lat_param_T import Lat_param_T
from apex.core.lib.utils import create_path
from apex.core.lib.util import collect_task
from apex.core.lib.dispatcher import make_submission
from apex.utils import sepline, get_task_type, handle_prop_suffix
from dflow.python import upload_packages
upload_packages.append(__file__)

lammps_task_type = ['deepmd', 'eam_alloy', 'meam', 'eam_fs', 'meam_spline', 'snap', 'gap', 'rann', 'mace']

def make_property_instance(parameters, inter_param):
    """
    Make an instance of Property
    """
    prop_type = parameters["type"]
    if prop_type == "eos":
        return EOS(parameters, inter_param)
    elif prop_type == "elastic":
        return Elastic(parameters, inter_param)
    elif prop_type == "vacancy":
        return Vacancy(parameters, inter_param)
    elif prop_type == "interstitial":
        return Interstitial(parameters, inter_param)
    elif prop_type == "surface":
        return Surface(parameters, inter_param)
    elif prop_type == "gamma":
        return Gamma(parameters, inter_param)
    elif prop_type == "phonon":
        return Phonon(parameters, inter_param)
    elif prop_type == "DecohesionEnergy":
        return DecohesionEnergy(parameters, inter_param)
    elif prop_type == "Lat_param_T":
        return Lat_param_T(parameters, inter_param)
    else:
        raise RuntimeError(f"unknown APEX type {prop_type}")


def make_property(confs, inter_param, property_list):
    # find all POSCARs and their name like mp-xxx
    # conf_dirs = glob.glob(confs)
    # conf_dirs.sort()
    conf_dirs = []
    for conf in confs:
        conf_dirs.extend(glob.glob(conf))
    conf_dirs = list(set(conf_dirs))
    conf_dirs.sort()
    for ii in conf_dirs:
        sepline(ch=ii, screen=True)
        path_to_equi = os.path.join(ii, "relaxation", "relax_task")
        try:
            structure_dict = loadfn(os.path.join(path_to_equi, "structure.json"))
        except FileNotFoundError:
            structure_dict = {}
        mismatch = structure_dict.get("mismatch", False)
        for jj in property_list:
            do_refine, suffix = handle_prop_suffix(jj)
            if not suffix:
                continue
            # generate working directory like mp-xxx/eos_00 if jj['type'] == 'eos'
            # handel the exception that the working directory exists
            # determine the suffix: from scratch or refine

            property_type = jj["type"]
            path_to_work = os.path.join(ii, property_type + "_" + suffix)
            skip_mismatch = jj.get("skip_mismatch", False)
            if mismatch and skip_mismatch:
                print("Skip mismatched structure")
                continue

            create_path(path_to_work)

            inter_param_prop = jj.get("cal_setting", {}).get("overwrite_interaction", inter_param)

            prop = make_property_instance(jj, inter_param_prop)
            task_list = prop.make_confs(path_to_work, path_to_equi, do_refine)

            for kk in task_list:
                poscar = os.path.join(kk, "POSCAR")
                inter = make_calculator(inter_param_prop, poscar)
                inter.make_potential_files(kk)
                inter.make_input_file(kk, prop.task_type(), prop.task_param())

            prop.post_process(task_list)  # generate same KPOINTS file for elastic when doing DFT


def worker(
    work_path,
    all_task,
    forward_common_files,
    forward_files,
    backward_files,
    mdata,
    inter_type,
    task_type,
):
    run_tasks = [os.path.basename(ii) for ii in all_task]
    machine = mdata.get("machine", None)
    resources = mdata.get("resources", None)
    command = mdata.get(f"{task_type}_run_command", mdata.get("run_command", None))
    group_size = mdata.get("group_size", 1)

    submission = make_submission(
        mdata_machine=machine,
        mdata_resources=resources,
        commands=[command],
        work_path=work_path,
        run_tasks=run_tasks,
        group_size=group_size,
        forward_common_files=forward_common_files,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog="outlog",
        errlog="errlog",
    )
    submission.run_submission()


def run_property(confs, inter_param, property_list, mdata):
    # find all POSCARs and their name like mp-xxx
    # conf_dirs = glob.glob(confs)
    # conf_dirs.sort()
    conf_dirs = []
    for conf in confs:
        conf_dirs.extend(glob.glob(conf))
    conf_dirs = list(set(conf_dirs))
    conf_dirs.sort()

    task_list = []
    work_path_list = []
    multiple_ret = []
    for ii in conf_dirs:
        sepline(ch=ii, screen=True)
        for jj in property_list:
            # determine the suffix: from scratch or refine
            # ...
            do_refine, suffix = handle_prop_suffix(jj)
            if not suffix:
                continue

            property_type = jj["type"]
            path_to_work = os.path.abspath(
                os.path.join(ii, property_type + "_" + suffix)
            )

            work_path_list.append(path_to_work)
            tmp_task_list = glob.glob(os.path.join(path_to_work, "task.[0-9]*[0-9]"))
            tmp_task_list.sort()
            task_list.append(tmp_task_list)

            inter_param_prop = jj.get("cal_setting", {}).get("overwrite_interaction", inter_param)

            # dispatch the tasks
            # POSCAR here is useless
            virtual_calculator = make_calculator(inter_param_prop, "POSCAR")
            forward_files = virtual_calculator.forward_files(property_type)
            forward_common_files = virtual_calculator.forward_common_files(
                property_type
            )
            backward_files = virtual_calculator.backward_files(property_type)
            #    backward_files += logs
            # ...
            task_type = get_task_type({"interaction": inter_param})
            inter_type = inter_param_prop["type"]
            work_path = path_to_work
            all_task = tmp_task_list
            run_tasks = collect_task(all_task, inter_type)
            if len(run_tasks) == 0:
                continue
            else:
                processes = len(run_tasks)
                pool = Pool(processes=processes)
                print("Submit job via %d processes" % processes)
                ret = pool.apply_async(
                    worker,
                    (
                        work_path,
                        all_task,
                        forward_common_files,
                        forward_files,
                        backward_files,
                        mdata,
                        inter_type,
                        task_type
                    )
                )
                multiple_ret.append(ret)
    pool.close()
    pool.join()
    for ii in range(len(multiple_ret)):
        if not multiple_ret[ii].successful():
            print("ERROR:", multiple_ret[ii].get())
            raise RuntimeError("Job %d is not successful!" % ii)
    print("%d jobs are finished" % len(multiple_ret))


def post_property(confs, inter_param, property_list):
    # find all POSCARs and their name like mp-xxx
    #    task_list = []
    # conf_dirs = glob.glob(confs)
    # conf_dirs.sort()
    conf_dirs = []
    for conf in confs:
        conf_dirs.extend(glob.glob(conf))
    conf_dirs = list(set(conf_dirs))
    conf_dirs.sort()
    for ii in conf_dirs:
        for jj in property_list:
            # determine the suffix: from scratch or refine
            # ...
            do_refine, suffix = handle_prop_suffix(jj)
            if not suffix:
                continue

            inter_param_prop = jj.get("cal_setting", {}).get("overwrite_interaction", inter_param)

            property_type = jj["type"]
            path_to_work = os.path.join(ii, property_type + "_" + suffix)
            prop = make_property_instance(jj, inter_param_prop)
            param_json = os.path.join(path_to_work, "param.json")
            param_dict = prop.parameter
            param_dict.setdefault("skip", False) # default of "skip" is False
            try:
                param_dict.pop("skip")
            except KeyError:
                pass
            dumpfn(param_dict, param_json)
            prop.compute(
                os.path.join(path_to_work, "result.json"),
                os.path.join(path_to_work, "result.out"),
                path_to_work
            )
