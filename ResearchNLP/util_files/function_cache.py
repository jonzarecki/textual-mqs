import cPickle as pkl
import json
import os

from ResearchNLP.util_files import file_util
from ResearchNLP.util_files.parallel_load import ParallelLoad
from config import CODE_DIR

caches_dicts = dict()
orig_caches_dicts = dict()


def func_cache(func):
    name = func.func_name

    def func_wrapper(*args, **kw_args):
        load_cache_in_parallel(func)
        orig_func_cache = orig_caches_dicts[name][1].obj if orig_caches_dicts[name][1].is_finished() else dict()
        curr_f_cache = caches_dicts[name][1]
        args_hash = hash((args, json.dumps(kw_args, sort_keys=True)))
        res = orig_func_cache.get(args_hash) or curr_f_cache.get(args_hash)  # saves a read from dict
        if res is None:  # didn't find in cache
            res = func(*args, **kw_args)
            curr_f_cache[args_hash] = res
        return res  # from cache or calculation
    func_wrapper.__name__ = func.func_name
    return func_wrapper


def load_cache_in_parallel(func):
    name = func.func_name
    from ResearchNLP import Constants as cn
    cache_path = CODE_DIR + "indexes/functions/" + name + "/" + cn.data_name + "_dict.pkl"
    if name not in orig_caches_dicts or orig_caches_dicts[name][0] != cache_path:
        if name in orig_caches_dicts or name in caches_dicts:  # old cache exists, save it
            save_cache(name)

        orig_caches_dicts[name] = [None, None]  # delete old caches
        caches_dicts[name] = [None, None]

        caches_dicts[name][0] = cache_path  # update curr path
        orig_caches_dicts[name][0] = cache_path

        caches_dicts[name][1] = dict()
        try:
            if os.path.exists(cache_path):
                orig_caches_dicts[name][1] = ParallelLoad(pkl_path=cache_path)
            else:
                orig_caches_dicts[name][1] = ParallelLoad(loading_fun=lambda: dict())
        except:
            orig_caches_dicts[name][1] = ParallelLoad(loading_fun=lambda: dict())

def force_cache_save(func, result, *args, **kw_args):
    name = func.func_name
    args_hash = hash((args, json.dumps(kw_args, sort_keys=True)))
    func_cache = caches_dicts[name][1]
    func_cache[args_hash] = result


def try_to_extract(func, *args, **kw_args):
    name = func.func_name
    args_hash = hash((args, json.dumps(kw_args, sort_keys=True)))
    func_cache = caches_dicts[name][1]
    return func_cache.get(args_hash)


def save_cache(key=None):
    # return  # not persistent
    def save_specific_cache(func_name):
        cache_path = caches_dicts[func_name][0]
        d1 = orig_caches_dicts[func_name][1].obj if func_name in orig_caches_dicts else dict()
        d2 = caches_dicts[func_name][1] if func_name in caches_dicts else dict()
        print "save old " + func_name + " cache"
        file_util.makedirs(os.path.dirname(cache_path))
        pkl.dump(_merge_two_dicts(d1, d2), open(cache_path, 'wb'))

    if key is not None:
        relevant_keys = [key]
    else:
        relevant_keys = caches_dicts.keys()
    for fnc_name in relevant_keys:
        save_specific_cache(fnc_name)


def merge_cache_dicts_from_parallel_runs(dicts_list):
    if len(dicts_list) == 1:
        return dicts_list[0]
    master_dict = dicts_list[0]
    for curr_d in dicts_list[1:]:
        for curr_key in master_dict:
            if curr_key not in curr_d:
                continue
            assert master_dict[curr_key][0] == curr_d[curr_key][0], "similar caches should have same paths"
            master_dict[curr_key][1] = _merge_two_dicts(master_dict[curr_key][1], curr_d[curr_key][1])
        for curr_key in curr_d:
            if curr_key not in master_dict:
                master_dict[curr_key] = curr_d[curr_key]
    global caches_dicts, orig_caches_dicts
    caches_dicts = master_dict
    # update orig_caches_dicts to have all entries
    for key in caches_dicts.keys():
        curr_path = caches_dicts[key][0]
        if key not in orig_caches_dicts or orig_caches_dicts[key][0] != curr_path:
            orig_caches_dicts[key] = [curr_path, None]  # delete old caches
            orig_caches_dicts[key][0] = curr_path  # update curr path
            try:
                if os.path.exists(curr_path):
                    orig_caches_dicts[key][1] = ParallelLoad(pkl_path=curr_path)
                else:
                    orig_caches_dicts[key][1] = ParallelLoad(loading_fun=lambda: dict())
            except:
                orig_caches_dicts[key][1] = ParallelLoad(loading_fun=lambda: dict())


def _merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z