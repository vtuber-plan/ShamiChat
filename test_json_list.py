from shami.data.jsonlist import JsonList

json_list = JsonList("temp_list", chunk_size=10, dir_chunk_num=10)

json_list.extend([str(i) for i in range(32)])