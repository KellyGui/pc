from asyncio import as_completed

import open3d as o3d
import piexif
import numpy as np
import time
import pyproj
import os
import glob
import geojson
import laspy
from osgeo import gdal
import rasterize
import rasterio as rio
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import rasterio

from rasterio.merge import merge

from rasterio.plot import show
import re

import os
import math





def getPOS(f,transformer):
    exif_dict = piexif.load(f)
    fullpath, name = os.path.split(f)
    long = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
    lat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
    alt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]

    du_long = long[0][0] / long[0][1] + long[1][0] / long[1][1] / 60 + long[2][0] / long[2][1] / 3600
    du_lat = lat[0][0] / lat[0][1] + lat[1][0] / lat[1][1] / 60 + lat[2][0] / lat[2][1] / 3600
    f_height = alt[0] / alt[1]

    
    x, y = transformer.transform(du_lat, du_long)
    return x,y,f_height



def rectifyPC(pcl1,pcl2):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl1)
    # pcd.paint_uniform_color([0.0, 1.0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcl2)
    # pcd2.paint_uniform_color([1.0, 0, 0])

    res = o3d.pipelines.registration.registration_icp(source=pcd, target=pcd2,
                                                      max_correspondence_distance=500,
                                                      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return res.transformation

# adapt to multi lines reconstruction
def getPicsPc(path,JPGfiles,espg):
    # JPGfiles = glob.glob(picFolder+"/*.*")
    # writefile = os.path.join(savepath,"odmcams.txt")
    # writefile2 = os.path.join(savepath,"picCams.txt")
    points = []
    p1=[]
    p2=[]

    with open(path) as f:
        gj = geojson.load(f)
    features = gj['features']
    for jpg in JPGfiles:
        nameJPG = os.path.basename(jpg)
        #去除中文及中文字符
        name_old = re.sub('[\u4e00-\u9fa5]','',nameJPG)
        # if (nameJPG.find('VIS') != -1):
        #     nameJPG = nameJPG.split('_')[2]
        transformer = pyproj.Transformer.from_crs(4326,espg)
        for feat in features:
            name = feat["properties"]["filename"]
            name_new =   re.sub('[\u4e00-\u9fa5]','',name)

            if (name_new == name_old):
                points.append(feat["properties"]["translation"])
                # wtxt = os.path.join(writefile, os.path.basename(jpg)[:-4]+'.txt')
                # with open(writefile, 'a') as ff:
                pp = feat["properties"]["translation"]
                p1.append(pp)
                x, y, h = getPOS(jpg,transformer)
                p2.append([x,y,h])
                break

    return np.asarray(p1),np.asarray(p2)
    

def computGPS(data_in,transformer):
    x,y = transformer.transform(data_in[0],data_in[1])    
    return x,y

def main(
    gps_points,
    dsm_out,
    clr_out=None,
    resolution=0.5,
    radius=1,
    sigma=None,
    roi=None,
):
    """
    Convert point cloud las to dsm tif
    """
    
    points = np.vstack(( gps_points[:,1],  gps_points[:,0]))
    values =  gps_points[:,2]
    values = values[np.newaxis, ...]
    # else:
    #         values = np.vstack((height, las.red, las.green, las.blue))
    valid = np.ones((1, points.shape[1]))

    # if roi is None:
    #     attrs = str(Path(cloud_in).with_suffix("")) + "_attrs.json"
    #     if Path(attrs).exists():
    #         with open(attrs, "r", encoding="utf-8") as attrs_reader:
    #             roi = json.load(attrs_reader)["attributes"]
    #     else:
    roi = {
                "xmin": resolution
                * ((np.amin(gps_points[:,1]) - resolution / 2) // resolution),
                "ymax": resolution
                * ((np.amax(gps_points[:,0]) + resolution / 2) // resolution),
                "xmax": resolution
                * ((np.amax(gps_points[:,1]) + resolution / 2) // resolution),
                "ymin": resolution
                * ((np.amin(gps_points[:,0]) - resolution / 2) // resolution),
            }

    roi["xstart"] = roi["xmin"]
    roi["ystart"] = roi["ymax"]
    roi["xsize"] = (roi["xmax"] - roi["xmin"]) / resolution
    roi["ysize"] = (roi["ymax"] - roi["ymin"]) / resolution

    if sigma is None:
        sigma = resolution

    # pylint: disable=c-extension-no-member
    out, mean, stdev, nb_pts_in_disc, nb_pts_in_cell = rasterize.pc_to_dsm(
        points,
        values,
        valid,
        roi["xstart"],
        roi["ystart"],
        int(roi["xsize"]),
        int(roi["ysize"]),
        resolution,
        radius,
        sigma,
    )

    # reshape data as a 2d grid.
    shape_out = (int(roi["ysize"]), int(roi["xsize"]))
    out = out.reshape(shape_out + (-1,))
    mean = mean.reshape(shape_out + (-1,))
    stdev = stdev.reshape(shape_out + (-1,))
    nb_pts_in_disc = nb_pts_in_disc.reshape(shape_out)
    nb_pts_in_cell = nb_pts_in_cell.reshape(shape_out)

    # save dsm
    # out: gaussian interpolation
    transform = Affine.translation(roi["xstart"], roi["ystart"])
    transform = transform * Affine.scale(resolution, -resolution)

    profile = DefaultGTiffProfile(
        count=1,
        dtype=out.dtype,
        width=roi["xsize"],
        height=roi["ysize"],
        transform=transform,
        nodata=np.nan,
    )

    with rio.open(dsm_out, "w", **profile) as dst:
        dst.write(out[..., 0], 1)

    if clr_out is not None:
        # clr: color r, g, b
        transform = Affine.translation(roi["xstart"], roi["ystart"])
        transform = transform * Affine.scale(resolution, -resolution)

        profile = DefaultGTiffProfile(
            count=3,
            dtype=out.dtype,
            width=roi["xsize"],
            height=roi["ysize"],
            transform=transform,
            nodata=np.nan,
        )

        with rio.open(clr_out, "w", **profile) as dst:
            for band in range(3):
                dst.write(out[..., band + 1], band + 1)

def checkMatch(R,pc1,pc2):
    count=0
    a11 = R[0,0]
    a12 = R[0,1]
    a13 = R[0,2]
   
    a21 = R[1,0]
    a22 = R[1,1]
    a23 = R[1,2]
    
    a31 = R[2,0]
    a32 = R[2,1]
    a33 = R[2,2]
    
    t1 = R[0,3]
    t2 = R[1,3]
    t3 = R[2,3]
    
    size=0
    for i, p in enumerate(pc1):
        tmpx = a11*p[0]+a12*p[1]+a13*p[2]+t1
        tmpy = a21*p[0]+a22*p[1]+a23*p[2]+t2
        tmpz = a31*p[0]+a32*p[1]+a33*p[2]+t3
        newp = [tmpx,tmpy,tmpz]
        dist = math.dist(newp,pc2[i])
        size += 1
        if(dist<10):
            count +=1

    if count/size>0.8:
        return True
    return False     

def  rectify(las,imageFiles,camjson,dsm_out,espg):
    #ICP
    pc1,pc2 = getPicsPc(camjson,imageFiles,espg)
    R = rectifyPC(pc1,pc2)

    # check goodmatch
    # if checkMatch(R,pc1,pc2)==False:
    #     print('bad result.')
    #     return 0
    
    with laspy.open(las) as fh:
        # print('Points from Header:', fh.header.point_count)
        data_las = fh.read()
        #rectify point cloud
        # data_las = laspy.read(las) # read a las file

        xyz = np.array(list(zip(data_las.x,data_las.y,data_las.z)))
        clr = np.array(list(zip(data_las.red/256.0,data_las.green/256.0, data_las.blue/256.0)))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(clr)
        pcd.transform(R)
        # print("transformation is ",R)
        points = np.asarray(pcd.points)
        data_las.x = points[:,0]
        data_las.y = points[:,1]
        data_las.z = points[:,2]
        #save rect point cloud in las format
        data_las.write(las[:-4] + "_rect.las")
        # o3d.io.write_point_cloud(las[:-4] + "_rect.ply", pcd)
        
        #convert to GPS
        transformer = pyproj.Transformer.from_crs(espg,4326)
        points = np.asarray(pcd.points)
        gps_points=[]
        for a in points:
            x,y = computGPS(a,transformer)
            gps_points.append([x,y,a[2]])
        
        gps_points = np.asarray(gps_points)

        #generate dsm
        main(
            gps_points ,
            dsm_out,
            clr_out=None,
            resolution=0.000007,
            radius=0,
            sigma=None,
            roi=None,
        )

# def mergeTif(file_list,dsm_out):
#     file_sorted = sorted(file_list,key = os.path.getmtime)
#     files_string = " ".join(file_sorted)

#     command = "gdal_merge.py -o "+dsm_out+" -of GTiff " + files_string

#     os.system(command)
def generatedsmUTM(ply,dsm_out):
    pcd = o3d.io.read_point_cloud(ply)
    points = np.asarray(pcd.points)
    main(
            points ,
            dsm_out,
            clr_out=None,
            resolution=0.5,
            radius=0,
            sigma=None,
            roi=None,
        )



def function_c(x1,x2):
    if np.isnan(x1) and not np.isnan(x2):
        result=x2
    elif np.isnan(x2):
        result=x1
    else:
        result=x2
    return result

def mergeTif(file_list,dsm_out):
    file_sorted = sorted(file_list,key = os.path.getmtime)

    lminx=[]
    lminy=[]
    lmaxx=[]
    lmaxy=[]
    
    for fp in file_list:
        im_Geotrans,cols,rows,im_data,im_datatype = read_tif(fp)
        minx = im_Geotrans[0]
        miny = im_Geotrans[3]
        maxx = im_Geotrans[0]+cols*im_Geotrans[1]
        maxy = miny+rows*im_Geotrans[5]
        c1 = round( (maxx-minx)/im_Geotrans[1])
        r1 =round((maxy-miny)/im_Geotrans[5])
        # print(r1,c1,rows,cols)
        lminx.append(minx)
        lminy.append(miny)
        lmaxx.append(maxx)
        lmaxy.append(maxy)
    
    mminx = min(lminx)
    mminy = max(lminy)
    mmaxx = max(lmaxx)
    mmaxy = min(lmaxy)
    # print(mminx,mminy,mmaxx,mmaxy)
    res=list(im_Geotrans)
    res[0]=mminx
    res[3]=mminy
    cols = round( (mmaxx-mminx)/im_Geotrans[1])
    rows = round((mmaxy-mminy)/im_Geotrans[5])
    print(rows,cols)
    vectorfunc = np.vectorize(function_c)
    #创建 底图
    d=np.full((rows,cols),np.nan)
    # d[np.isnan(d)]=np.nan
    for fp in file_sorted:
        geotrans,c,r,im_data, im_datatype = read_tif(fp)
        delta_c = int((geotrans[0]-mminx)/im_Geotrans[1])
        delta_r = int((geotrans[3]-mminy)/im_Geotrans[5])

        b_mask_boolean = np.isnan(im_data)
        d[delta_r:r+delta_r,delta_c:c+delta_c] = vectorfunc(d[delta_r:r+delta_r,delta_c:c+delta_c],im_data)
        # np.nan_to_num(im_data, copy=False,nan=-999)
        # d[delta_r:r+delta_r,delta_c:c+delta_c] =  d[delta_r:r+delta_r,delta_c:c+delta_c]*b_mask_boolean + im_data * ~b_mask_boolean
    # dsm_path = os.path.dirname(os.path.dirname(file_list[0]))+"/combine.tif"
    write_tif(dsm_out,d,tuple(res),'',im_datatype)

def read_tif(path):
    dataset = gdal.Open(path)
    # print(dataset.GetDescription())#数据描述
    band1 = dataset.GetRasterBand(1)
    im_datatype = band1.DataType

    cols = dataset.RasterXSize#图像长度
    rows = (dataset.RasterYSize)#图像宽度
    im_proj = (dataset.GetProjection())#读取投影
    im_Geotrans = (dataset.GetGeoTransform())#读取仿射变换
    im_data = dataset.ReadAsArray(0, 0, cols, rows)#转为numpy格式
    # print(im_Geotrans[0],im_Geotrans[3])
    # minx = im_Geotrans[0]
    # miny = im_Geotrans[3]
    # maxx = im_Geotrans[0]+cols*im_Geotrans[1]
    # maxy = miny+rows*im_Geotrans[5]
    #im_data[im_data > 0] = 1 #除0以外都等于1
    del dataset
    return im_Geotrans,cols,rows,im_data,im_datatype

def write_tif(newpath,im_data,im_Geotrans,im_proj,datatype):
    if len(im_data.shape)==3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del new_dataset



if __name__ == '__main__':
    las = '/server-w/3Dreconstruction/虎山/013545/odm_georeferencing/odm_georeferenced_model.laz'
    image_path = ['/server-w/3Dreconstruction/虎山/B01/images', '/server-w/3Dreconstruction/虎山/C01_45/images']
    imageFiles = []
    for path in image_path:
        imageFiles.extend(glob.glob(path+"/*.*"))
    camjson = '/server-w/3Dreconstruction/虎山/013545/odm_report/shots.geojson'
    save_dsm = '/server-w/3Dreconstruction/虎山/013545/manual.tif'
    # pc_out = '/home/a62/GMP/open3d/jiagnshanshimen/10/10-georeferenced_model_rect.las'
    # dsm_out = '/server-w/3Dreconstruction/杭州视频建模35/B01-35/'+ time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))+'.tif'
    t1 = time.time()
    # tif = '/server-w/3Dreconstruction/万田/02/5ad1442c-44a7-4d87-b6be-2d7537c257b1.tif'
    # read_tif(tif)
    # rectply='/server-w/3Dreconstruction/杭州视频建模35/B01-35/manual_rect.ply'
    # generatedsmUTM(rectply,dsm_out)
    rectify(las,imageFiles,camjson,save_dsm,32650)
    # file_list = glob.glob('/server-w/3Dreconstruction/无锡视频建模35'+'/**/*.tif')
  
    # mergeTif(file_list, dsm_out)
    t2= time.time()
    print(t2-t1)
    






