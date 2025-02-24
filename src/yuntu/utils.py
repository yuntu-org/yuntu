"""Yuntu utilities."""
import importlib.util
from importlib import import_module

import os
import io
import tempfile
import subprocess
from contextlib import contextmanager

import requests
from tqdm import tqdm


TMP_DIR = os.path.join(tempfile.gettempdir(), 'yuntu')


@contextmanager
def tmp_file(basename):
    filename = os.path.join(TMP_DIR, basename)

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'wb') as tmpfile:
        yield filename, tmpfile

def download_file_alt(r):
    buffer = io.BytesIO()
    chunk_size = 1024
    for chunk in r.iter_content(chunk_size=chunk_size):
        if chunk:  
            buffer.write(chunk)
    buffer.seek(0)
    return buffer

def download_file(url):
    r = requests.get(url, stream=True)
    if "Content-Length" in r.headers:
        buffer = io.BytesIO()
        file_size = int(r.headers['Content-Length'])
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)
        iterable = tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=num_bars,
            desc=url,
            leave=True,
            unit='KB')
    
        for chunk in iterable:
            buffer.write(chunk)
        buffer.seek(0)
        return buffer
    else:
        return download_file_alt(r)


def scp_file(src, dest):
    filename = os.path.join(TMP_DIR, dest)
    print(f'Downloading file {src}...', end='')
    subprocess.run(['scp', src, filename], check=True)
    print(' done.')
    return filename

def load_module_object(object_name):
    name_arr = object_name.split(".")
    if len(name_arr) > 1:
        meth = import_module(name_arr[0])
        for i in range(1,len(name_arr)):
            last = getattr(meth, name_arr[i])
            meth = last

        return meth

    else:
        raise ValueError("Only module specified.")

def load_module_object_from_file(path, object_name):
    spec = importlib.util.spec_from_file_location(object_name, path)
    modl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modl)

    name_arr = object_name.split(".")

    if len(name_arr) > 1:
        meth = getattr(modl, name_arr[0])
        for i in range(1, len(name_arr)):
            last = getattr(meth, name_arr[i])
            meth = last

        return meth

    else:
        return getattr(modl, object_name)

def module_object(module_config):
    if "path" in module_config:
        if module_config["path"] is not None:
            return load_module_object_from_file(module_config["path"],
                                                module_config["object_name"])
    return load_module_object(module_config["object_name"])
