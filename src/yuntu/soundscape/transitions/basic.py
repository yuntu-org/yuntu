
"""Transitions for basic usage."""
import os
import math
import shutil
import copy
import numpy as np
import pandas as pd
import requests
import dask.dataframe as dd
import dask.bag as db
from pony.orm import db_session

from yuntu.utils import module_object
from yuntu.core.audio.audio import Audio, MEDIA_INFO_FIELDS
from yuntu.core.database.mixins.utils import pg_create_db
from yuntu.collection.methods import collection
from yuntu.core.pipeline.places import *
from yuntu.core.pipeline.transitions.decorators import transition
from yuntu.soundscape.hashers.base import Hasher
from yuntu.soundscape.utils import absolute_timing


def get_fragment_size(col_config, query, limit=None, offset=0):
    col = collection(**col_config)
    if limit is None:
        query_slice = slice(offset, None)
    else:
        query_slice = slice(offset, offset + limit)

    with db_session:
        fragment_length = len(col.recordings(query=query).order_by(lambda r: r.id)[query_slice])

    col.db_manager.db.disconnect()
    return fragment_length

def insert_datastore(dstore_config, col_config):
    dstore_class = module_object(dstore_config["module"])
    dstore_kwargs = dstore_config["kwargs"]

    datastore = dstore_class(**dstore_kwargs)
    col = collection(**col_config)

    with db_session:
        datastore_id, recording_inserts, annotation_inserts = datastore.insert_into(col)

    col.db_manager.db.disconnect()

    return {"datastore_record": datastore_id,
            "recording_inserts": recording_inserts,
            "annotation_inserts": annotation_inserts}

def apply_absolute_time(row, time_col):
    new_row = {}
    new_row["abs_start_time"] = absolute_timing(row[time_col], row["start_time"])
    new_row["abs_end_time"] = absolute_timing(row[time_col], row["end_time"])

    return pd.Series(new_row)

def unash_hash(row, unhash, hash_col):
    uhash = unhash(row[hash_col])
    new_row = {}
    new_row[f"{hash_col}_time"] = uhash
    return pd.Series(new_row)

@transition(name='add_hash', outputs=["hashed_soundscape"],
            keep=True, persist=True, is_output=True,
            signature=((DaskDataFramePlace, PickleablePlace, ScalarPlace),
                       (DaskDataFramePlace, )))
def add_hash(dataframe, hasher_config, out_name="xhash"):
    hasher_class = module_object(hasher_config["module"])
    hasher_kwargs = hasher_config["kwargs"]
    hasher = hasher_class(**hasher_kwargs)

    if not hasher.validate(dataframe):
        str_cols = str(hasher.columns)
        message = ("Input dataframe is incompatible with hasher."
                   f"Missing column inputs. Hasher needs: {str_cols} ")
        raise ValueError(message)

    meta = [(out_name, hasher.dtype)]
    result = dataframe.apply(hasher, out_name=out_name, meta=meta, axis=1)
    dataframe[out_name] = result[out_name]

    return dataframe


@transition(name='add_absoute_time', outputs=["absolute_timed_soundscape"],
            keep=True, persist=True, is_output=True,
            signature=((DaskDataFramePlace, ScalarPlace),
                       (DaskDataFramePlace, )))
def add_absoute_time(dataframe, time_col):
    meta = [("abs_start_time", np.dtype('datetime64[ns]')), ("abs_end_time", np.dtype('datetime64[ns]'))]
    result = dataframe.apply(apply_absolute_time, time_col=time_col, meta=meta, axis=1)
    dataframe["abs_start_time"] = result["abs_start_time"]
    dataframe["abs_end_time"] = result["abs_end_time"]

    return dataframe

@transition(name='as_dd', outputs=["recordings_dd"],
            signature=((PandasDataFramePlace, ScalarPlace),
                       (DaskDataFramePlace,)))
def as_dd(pd_dataframe, npartitions):
    """Transform audio dataframe to a dask dataframe for computations."""
    dask_dataframe = dd.from_pandas(pd_dataframe,
                                    npartitions=npartitions,
                                    name="as_dd")
    return dask_dataframe


@transition(name="source_partition", outputs=["datastore_configs"],
            signature=((DynamicPlace, DynamicPlace, ScalarPlace), (DynamicPlace,)))
def source_partition(datastore_config, rest_auth, npartitions=1):
    metadata_url = datastore_config["kwargs"]["metadata_url"]
    url = metadata_url+"&page_size=1"
    item_count = requests.get(url, auth=rest_auth).json()["count"]
    page_size = datastore_config["kwargs"]["page_size"]
    total_pages = math.ceil(float(item_count)/float(page_size))
    partition_size = math.ceil(total_pages/npartitions)

    partitions = []
    for n in range(npartitions):
        page_start = n*partition_size+1
        page_end = (n+1)*partition_size
        part_config = copy.deepcopy(datastore_config)
        part_config["kwargs"]["page_start"] = page_start
        part_config["kwargs"]["page_end"] = page_end
        part_config["kwargs"]["auth"] = rest_auth
        partitions.append(part_config)

    return partitions


@transition(name="get_partitions", outputs=["partitions"],
            signature=((DictPlace, DynamicPlace, ScalarPlace, ScalarPlace, ScalarPlace), (DynamicPlace,)))
def get_partitions(col_config, query, npartitions=1, limit=None, offset=0):
    length = get_fragment_size(col_config, query, limit=limit, offset=offset)
    if length == 0:
        raise ValueError("Collection has no data. Populate collection first.")

    psize = int(np.floor(length / npartitions))
    psize = min(length, max(psize, 1))

    partitions = []
    offset_length = offset+length
    for ind in range(offset, offset_length, psize):
        ioffset = ind
        ilimit = min(psize, offset_length - ioffset)

        stop = False
        if offset_length - (ioffset + ilimit) < 30:
            ilimit = offset_length - ioffset
            stop = True

        partitions.append({"query": query, "limit": ilimit, "offset": ioffset})

        if stop:
            break


    return db.from_sequence(partitions, npartitions=len(partitions))


@transition(name="init_write_dir", outputs=["dir_exists"],
            signature=((DictPlace, DynamicPlace), (DynamicPlace,)))
def init_write_dir(write_config, overwrite=False):
    """Initialize output directory"""

    if os.path.exists(write_config["write_dir"]) and overwrite:
        shutil.rmtree(write_config["write_dir"], ignore_errors=True)
    if not os.path.exists(write_config["write_dir"]):
        os.makedirs(write_config["write_dir"])
    return True


@transition(name="pg_init_database", outputs=["col_config"],
            signature=((DictPlace, DynamicPlace), (DictPlace,)))
def pg_init_database(init_config, admin_config):
    pg_create_db(init_config["db_config"]["config"],
                 admin_user=admin_config["admin_user"],
                 admin_password=admin_config["admin_password"],
                 admin_db=admin_config["admin_db"])

    col = collection(**init_config)
    col.db_manager.db.disconnect()

    return init_config


@transition(name="load_datastores", outputs=["insert_results"], persist=True,
            signature=((DictPlace, DynamicPlace), (DaskDataFramePlace,)))
def load_datastores(col_config, dstore_configs):
    dstore_bag = db.from_sequence(dstore_configs, npartitions=len(dstore_configs))
    inserted = dstore_bag.map(insert_datastore, col_config=col_config)

    meta = [('datastore_record', np.dtype('int')),
            ('recording_inserts', np.dtype('int')),
            ('annotation_inserts', np.dtype('int'))]

    return inserted.to_dataframe(meta=meta)


@transition(name="bag_dataframe", outputs=["dataframe_bag"], persist=False,
            signature=((PandasDataFramePlace, ScalarPlace), (DynamicPlace,)))
def bag_dataframe(dataframe, npartitions):
    """Transform dataframe to dict bag."""
    if dataframe.empty:
        raise ValueError("Dataframe has no data.")
    total = dataframe.shape[0]
    size = int(np.floor(float(total)/float(npartitions)))
    if size <= 0:
        raise ValueError(f"Too many partitions. Max is {total} for this dataframe.")

    dict_dataframe = []
    for i in range(npartitions):
        start = i*size
        end = min((i+1)*size, total)
        if i == npartitions -1:
            end = total
        dict_dataframe.append(dataframe.iloc[start : end].to_dict(orient="records"))
        
    return db.from_sequence(dict_dataframe, npartitions=npartitions)

