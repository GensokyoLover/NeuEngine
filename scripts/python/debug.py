import json
import os
import pickle
import zstandard as zstd
import pyexr
file_list = os.listdir(r"H:\Falcor\media\inv_rendering_scenes\bunny_ref_nobunny_roughnesscorrect\buckets/")
file_list2 = os.listdir(r"H:\Falcor\media\inv_rendering_scenes\bunny_ref_nobunny_roughnesscorrect\buckets_1/")
## load xxx.pkl.zst
for file in file_list2:
    with open(r"H:\Falcor\media\inv_rendering_scenes\bunny_ref_nobunny_roughnesscorrect\buckets/" + file, "rb") as f:
        compressed_data = f.read()
    decompressed_data = zstd.decompress(compressed_data)
    sample = pickle.loads(decompressed_data)
    with open(r"H:\Falcor\media\inv_rendering_scenes\bunny_ref_nobunny_roughnesscorrect\buckets_1/" + file, "rb") as f:
        compressed_data = f.read()
    decompressed_data = zstd.decompress(compressed_data)
    sample2 = pickle.loads(decompressed_data)

    print(1)
    for i in range(3):
        pyexr.write(r"H:/{}_sampleuvi.exr".format(i), sample["reflect_uvi"][:,i,:].reshape(256,256,3))
        pyexr.write(r"H:/{}_sample2uvi.exr".format(i), sample2["reflect_uvi"][:,i,:].reshape(256,256,3))
    pyexr.write(r"H:/{}_acc.exr".format(i), sample["AccumulatePassoutput"].reshape(256,256,3))
    pyexr.write(r"H:/{}_acc2.exr".format(i), sample2["AccumulatePassoutput"].reshape(256,256,3))

    