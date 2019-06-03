from pymongo import MongoClient
from bson.objectid import ObjectId
from collections import OrderedDict
from yuntu.core.datastore.utils import hashDict

def datastoreGetSpec(ds):
    dSpec = {}
    dSpec["hash"] = ds.getHash()
    dSpec["type"] = ds.getType()
    dSpec["conf"] = ds.getConf()
    dSpec["metadata"] = ds.getMetadata()

    return dSpec

def datastoreGetType(ds):
    return ds.inputSpec["type"]

def datastoreGetConf(ds):
    dConf = {}
    for key in ["host","datastore","target","filter","fields","ukey"]:
        dConf[key] = ds.inputSpec["conf"][key]

    return dConf

def datastoreGetMetadata(ds):
    return ds.inputSpec["metadata"]

def datastoreGetHash(ds):
    formatedConf = ds.getConf()

    return hashDict(formatedConf)

def datastoreMongodbGetData(ds):
    def f(dsSpec):
        dsConf = dsSpec["conf"]
        client = MongoClient(dsConf["host"],maxPoolSize = 30)
        mDb = client[dsConf["datastore"]]
        collection = mDb[dsConf["target"]]

        if isinstance(dsConf["filter"],list):
            for rId in dsConf["filter"]:
                obj = collection.find_one({"_id":ObjectId(rId)})
                fkey = str(obj[dsConf["ukey"]])
                obj[dsConf["ukey"]] = fkey
                yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}
        else:
            for obj in collection.find(dsConf["filter"],dsConf["fields"]):
                fkey = str(obj[dsConf["ukey"]])
                obj[dsConf["ukey"]] = fkey
                yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}

    return f(ds.getSpec())

def datastoreAudioMothGetData(ds):
    def f(dsSpec):
        dsConf = dsSpec["conf"]
        dataDir = dsConf["dataDir"]
        allFiles = []
        for filename in os.listdir(dsConf["dataDir"]):
            if filename.endswith(".wav") or filename.endswith(".WAV"): 
                allFiles.append(filename)

        for i in range(len(allFiles)):
            fkey = allFiles[i]
            obj = {}

            tArr = os.path.splitext(fkey)[0].split("_")
            year = tArr[0][0:4]
            month = tArr[0][4:6]
            day = tArr[0][6:]
            hour = tArr[1][0:2]
            minute = tArr[1][2:4]
            second = tArr[1][4:]

            obj["time"] = day+"-"+month+"-"+year+" "+hour+":"+minute+":"+second
            obj["tZone"] = "UTC"
            
            yield {"datastore":dsSpec, "source":{"fkey":fkey},"metadata":obj}

    return f(ds.getSpec())

def datastoreDirectGetData(ds):
    def f(dsSpec,dataArr):
        for i in range(len(dataArr)):
            yield  {"datastore":dsSpec,"source":{"fkey":i},"metadata":dataArr[i]}

    return f(ds.getSpec(),ds.dataArr)


