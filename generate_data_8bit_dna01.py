import importlib
import numpy as np
import concurrent
import numpy as np
import pandas as pd
import imageio
import math
import multiprocessing
#from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
#from aicsfiles import FileManagementSystem # >=5.0.0.dev12
#from nuc_morph_analysis.preprocessing.load_data import load_data
from pathlib import Path
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.writers.two_d_writer import TwoDWriter
from skimage.measure import regionprops, label
from skimage import exposure, img_as_ubyte
from skimage import io

df_path  = "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/manifest_for_quilt.csv"

df = pd.read_csv(df_path)
#df1 = df.head()

# prepare file path
parent_path = Path("/allen/aics/assay-dev/users/Suraj/PCNA_Classification/data_max_masked_im0lb1/")
G1_path = parent_path / Path("G1")
G2_path = parent_path / Path("G2")
eS_path = parent_path / Path("eS")
mS_path = parent_path / Path("mS")
mSlS_path = parent_path / Path("mSlS")
eSmS_path = parent_path / Path("eSmS")
lS_path = parent_path / Path("lS")
lSG2_path = parent_path / Path("lSG2")
M_path = parent_path / Path("M")

for row in df.itertuples():
    reader_raw = AICSImage(df.at[row[0], "crop_raw"])
    reader_seg = AICSImage(df.at[row[0], "crop_seg"])
    
    vol_raw = reader_raw.get_image_data("ZYX", C=0, T=0)
    vol_seg = reader_seg.get_image_data("ZYX", C=1, T=0)

    #vol_raw = img_as_ubyte(exposure.rescale_intensity(vol_raw))
    vol_raw = (vol_raw / vol_raw.max()) * 255
    vol_raw = np.uint8(vol_raw)

    vol_raw_mip = np.max(vol_raw, axis=0)
    tt = np.argmax(np.sum(vol_seg,axis=(1,2)))
    vol_raw_max = vol_raw[tt,:,:]

    vol_seg_max = vol_seg[tt,:,:]
    vol_seg_max = vol_seg_max > 0 
    label_seg_max = label(vol_seg_max)
    props = regionprops(label_seg_max)

    leng = props[0].bbox[2] - props[0].bbox[0]
    widt = props[0].bbox[3] - props[0].bbox[1]

    adj_leng = math.floor(leng/4)
    adj_widt = math.floor(widt/4)

    bbox_new = (props[0].bbox[0] - adj_leng, props[0].bbox[1] - adj_widt,
            props[0].bbox[2] + adj_leng, props[0].bbox[3] + adj_widt)

    mask_base = np.zeros(vol_raw_max.shape)
    mask_base[bbox_new[0]:bbox_new[2], bbox_new[1]:bbox_new[3]] = 1

    #vol_raw_mip_masked_bb2 = vol_raw_mip * mask_base
    #vol_raw_max_masked_bb2 = vol_raw_max * mask_base

    vol_raw_mip_masked = vol_raw_mip * vol_seg_max
    vol_raw_max_masked = vol_raw_max * vol_seg_max

    target_name_raw = row[1] + '.tiff' #row[8].split('/')[-1]
    target_name_seg = row[1] + '_seg.tiff' #row[10].split('/')[-1]
    #target_name_mip = row[1] + '_np_mip.png' #row[8].split('/')[-1]
    target_name_max = row[1] + '_np_max.png'
    #target_name_mip_masked_base = row[1] + '_mip_masked_base2.png'
    target_name_max_masked_base = row[1] + '_max_masked_base2.png'
    #target_name_mip_masked_bb2 = row[1] + '_mip_masked_bb3.png'
    #target_name_max_masked_bb2 = row[1] + '_max_masked_bb3.png'


    if df.at[row[0], "cell_stage_fine"] == "G1":
        target_path_raw = G1_path / target_name_raw
        target_path_seg = G1_path / target_name_seg
        #target_path_mip = G1_path / target_name_mip
        target_path_max = G1_path / target_name_max
        #target_path_mip_masked_base = G1_path / target_name_mip_masked_base
        target_path_max_masked_base = G1_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = G1_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = G1_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "G2":
        target_path_raw = G2_path / target_name_raw
        target_path_seg = G2_path / target_name_seg
        #target_path_mip = G2_path / target_name_mip
        target_path_max = G2_path / target_name_max
        #target_path_mip_masked_base = G2_path / target_name_mip_masked_base
        target_path_max_masked_base = G2_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = G2_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = G2_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "earlyS":
        target_path_raw = eS_path / target_name_raw
        target_path_seg = eS_path / target_name_seg
        #target_path_mip = eS_path / target_name_mip
        target_path_max = eS_path / target_name_max
        #target_path_mip_masked_base = eS_path / target_name_mip_masked_base
        target_path_max_masked_base = eS_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = eS_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = eS_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "midS":
        target_path_raw = mS_path / target_name_raw
        target_path_seg = mS_path / target_name_seg
        #target_path_mip = mS_path / target_name_mip
        target_path_max = mS_path / target_name_max
        #target_path_mip_masked_base = mS_path / target_name_mip_masked_base
        target_path_max_masked_base = mS_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = mS_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = mS_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "midS-lateS":
        target_path_raw = mSlS_path / target_name_raw
        target_path_seg = mSlS_path / target_name_seg
        #target_path_mip = mSlS_path / target_name_mip
        target_path_max = mSlS_path / target_name_max
        #target_path_mip_masked_base = mSlS_path / target_name_mip_masked_base
        target_path_max_masked_base = mSlS_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = mSlS_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = mSlS_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "earlyS-midS":
        target_path_raw = eSmS_path / target_name_raw
        target_path_seg = eSmS_path / target_name_seg
        #target_path_mip = eSmS_path / target_name_mip
        target_path_max = eSmS_path / target_name_max
        #target_path_mip_masked_base = eSmS_path / target_name_mip_masked_base
        target_path_max_masked_base = eSmS_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = eSmS_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = eSmS_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "lateS":
        target_path_raw = lS_path / target_name_raw
        target_path_seg = lS_path / target_name_seg
        #target_path_mip = lS_path / target_name_mip
        target_path_max = lS_path / target_name_max
        #target_path_mip_masked_base = lS_path / target_name_mip_masked_base
        target_path_max_masked_base = lS_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = lS_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = lS_path / target_name_max_masked_bb2
    elif df.at[row[0], "cell_stage_fine"] == "lateS-G2":
        target_path_raw = lSG2_path / target_name_raw
        target_path_seg = lSG2_path / target_name_seg
        #target_path_mip = lSG2_path / target_name_mip
        target_path_max = lSG2_path / target_name_max
        #target_path_mip_masked_base = lSG2_path / target_name_mip_masked_base
        target_path_max_masked_base = lSG2_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = lSG2_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = lSG2_path / target_name_max_masked_bb2
    else:
        target_path_raw = M_path / target_name_raw
        target_path_seg = M_path / target_name_seg
        #target_path_mip = M_path / target_name_mip
        target_path_max = M_path / target_name_max
        #target_path_mip_masked_base = M_path / target_name_mip_masked_base
        target_path_max_masked_base = M_path / target_name_max_masked_base
        #target_path_mip_masked_bb2 = M_path / target_name_mip_masked_bb2
        #target_path_max_masked_bb2 = M_path / target_name_max_masked_bb2

    #OmeTiffWriter.save(vol_raw, target_path_raw, dim_order="ZYX")
    #OmeTiffWriter.save(vol_seg, target_path_seg, dim_order="ZYX")
    #TwoDWriter.save(vol_raw_mip, target_path_mip)
    #TwoDWriter.save(vol_raw_max, target_path_max)
    #TwoDWriter.save(vol_raw_mip_masked, target_path_mip_masked_base)
    TwoDWriter.save(vol_raw_max_masked, target_path_max_masked_base)
    #TwoDWriter.save(vol_raw_mip_masked_bb2, target_path_mip_masked_bb2)
    #TwoDWriter.save(vol_raw_max_masked_bb2, target_path_max_masked_bb2)
    #io.imsave(vol_raw_mip, target_path_mip)
    #io.imsave(vol_raw_max, target_path_max)
    #io.imsave(vol_raw_mip_masked, target_path_mip_masked_base)
    #io.imsave(vol_raw_max_masked, target_path_max_masked_base)
    #io.imsave(vol_raw_mip_masked_bb2, target_path_mip_masked_bb2)
    #io.imsave(vol_raw_max_masked_bb2, target_path_max_masked_bb2)

    #print(target_path_raw)
    #print(df1.at[row[0], "is_outlier"])    
