import gzip
import json
import os
from typing import Any, Dict, List, Optional

def write_jsonl_dataset(path: str, dataset: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def read_jsonl_dataset(path: str):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def write_jsonl_gz_dataset(path: str, dataset: List[Dict]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def read_jsonl_gz_dataset(path: str):
    dataset = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def read_dataset(path: str):
    if path.endswith(".jsonl"):
        return read_jsonl_dataset(path)
    elif path.endswith(".jsonl.gz"):
        return read_jsonl_gz_dataset(path)
    else:
        raise Exception("Unsupported File Extension.")

def write_dataset(path: str, dataset: List[Dict]):
    if path.endswith(".jsonl"):
        write_jsonl_dataset(path, dataset)
    elif path.endswith(".jsonl.gz"):
        write_jsonl_gz_dataset(path, dataset)
    else:
        raise Exception("Unsupported File Extension.")

class JsonListChunk(object):
    def __init__(self, root_path: str,
                    chunk_id: int,
                    chunk_size: int = 1000,
                    dir_chunk_num: int = 1000,
                    zip_algo: Optional[str] = None
                ) -> None:
        self.root_path = root_path
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.dir_chunk_num = dir_chunk_num
        self.zip_algo = zip_algo

        self.sub_dir_path = os.path.join(self.root_path, f'{chunk_id // self.dir_chunk_num:06d}')
        if not os.path.exists(self.sub_dir_path):
            os.makedirs(self.sub_dir_path)
        self.ext = self.get_ext(self.zip_algo)
        self.file_path = os.path.join(self.sub_dir_path, f'{chunk_id:06d}.{self.ext}')

        self.data: Optional[List[Any]] = None
        if not os.path.exists(self.file_path):
            self.data = []
            self.release()
        
    
    def get_ext(self, zip_algo: Optional[str]):
        if zip_algo is None:
            return "jsonl"
        elif zip_algo in ["gz", "gzip"]:
            return "jsonl.gz"
        else:
            raise Exception("Unsupported zip method.")
    
    def _write_chunk(self, chunk_data: List[Any]):
        if not os.path.exists(self.sub_dir_path):
            os.makedirs(self.sub_dir_path)

        write_dataset(self.file_path, chunk_data)
        return self.file_path
    
    def _read_chunk(self) -> List[Any]:
        if not os.path.exists(self.sub_dir_path):
            raise Exception("Cannot find sub dir.")

        return read_dataset(self.file_path)
    
    def load(self):
        if self.data is not None:
            return
        self.data = self._read_chunk()

    def release(self):
        if self.data is None:
            return
        self._write_chunk(self.data)
        self.data = None
    
    def flush(self):
        if self.data is None:
            return
        self._write_chunk(self.data)
    
    def is_memory(self) -> bool:
        return self.data is not None
    
    def append(self, value):
        self.load()
        self.data.append(value)

    def is_full(self) -> bool:
        self.load()
        return len(self.data) >= self.chunk_size
    
    def __getitem__(self, index: int):
        return self.data[index]

    def __setitem__(self, index: int, value: Any):
        self.data[index] = value

    def __len__(self) -> int:
        self.load()
        return len(self.data)

class JsonList(object):
    def __init__(self, root_path: str,
                    chunk_size: int = 1000,
                    dir_chunk_num: int = 1000,
                    zip_algo: Optional[str] = None
                ) -> None:
        self.root_path = root_path
        self.meta_path = os.path.join(self.root_path, "meta.json")

        if os.path.exists(root_path):
            if not os.path.exists(self.meta_path):
                raise Exception("meta file missing.")
            # load
            self.meta = self.read_meta(self.meta_path)
        else:
            # create
            os.makedirs(self.root_path)
            self.meta = {
                "version": "0.0.1",
                "chunk_size": chunk_size,
                "dir_chunk_num": dir_chunk_num,
                "zip_algo": zip_algo,
                "chunk_num": 0,
                "length": 0
            }
            self.write_meta(self.meta_path, self.meta)
        
        self.chunks: List[Optional[JsonListChunk]] = [
            JsonListChunk(
                self.root_path,
                chunk_id,
                self.meta["chunk_size"],
                self.meta["dir_chunk_num"],
                self.meta["zip_algo"]
            )
            for chunk_id in range(self.meta["chunk_num"])
        ]
        
    def read_meta(self, meta_path: str):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.loads(f.read())
        return meta

    def write_meta(self, meta_path: str, meta):
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta))
    
    def new_chunk(self):
        new_chunk_id = self.meta["chunk_num"]
        self.chunks.append(
            JsonListChunk(
                self.root_path,
                new_chunk_id,
                self.meta["chunk_size"],
                self.meta["dir_chunk_num"],
                self.meta["zip_algo"]
            )
        )
        self.meta["chunk_num"] += 1
        return self.chunks[-1]
    
    def append(self, item):
        if self.meta["chunk_num"] == 0:
            selected_chunk = self.new_chunk()
        else:
            last_chunk = self.chunks[-1]
            if len(last_chunk) == self.meta["chunk_size"]:
                last_chunk.flush()
                selected_chunk = self.new_chunk()
            else:
                selected_chunk = last_chunk
        
        selected_chunk.append(item)
        selected_chunk.flush()
        self.write_meta(self.meta_path, self.meta)

    def extend(self, items: List[Any]):
        for item in items:
            if self.meta["chunk_num"] == 0:
                selected_chunk = self.new_chunk()
            else:
                last_chunk = self.chunks[-1]
                if len(last_chunk) == self.meta["chunk_size"]:
                    last_chunk.flush()
                    selected_chunk = self.new_chunk()
                else:
                    selected_chunk = last_chunk
            selected_chunk.append(item)
            self.meta["length"] += 1
        selected_chunk.flush()
        self.write_meta(self.meta_path, self.meta)

    def get_item_single(self, index: int):
        chunk_id = index // self.meta["chunk_size"]
        chunk = read_dataset(chunk_id)
        chunk_index = index % self.meta["chunk_size"]
        return chunk[chunk_index]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item_single(index)
        elif isinstance(index, slice) or isinstance(index, range):
            ret = []
            i = index.start
            while i < index.stop:
                ret.append(self.get_item_single(i))
                i += index.step
            return ret
        else:
            raise Exception("Unsupported index type.")

    def __len__(self) -> int:
        return self.meta["length"]