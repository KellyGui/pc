# coding:utf-8
import glob
import os

import cv2
import PIL
import numpy as np
import piexif
from tqdm import tqdm
import time
from datetime import datetime
import math
import queue

import pyexiv2
import pyexiv2
from moviepy.editor import *
import glob

# choosepath = '/home/a62/GMP/open3d/jiangxi/B05/chosen'


def concateVideo(path):
        save_path = os.path.join(path,"all.mp4")
        if os.path.exists(save_path):
            return save_path
        videos = sorted(glob.glob(path+"/*.mp4"))
        clips = []
        for video in videos:
            tmp = VideoFileClip(video)
            clips.append(tmp)
        final_clip = concatenate_videoclips(clips)
        
        final_clip.to_videofile(save_path,fps=clips[0].fps, remove_temp=False)
        return save_path

class VideoProcess:
    """
        Initializes a new instance of the class.
        Args:
            videopath (str): The path to the video file or folder containing multiple video files.
            imageFolder (str): The path to the folder containing the images.
            savePath (str): The path to the folder where the output will be saved.
        Returns:
            None
    """
    def __init__(self, videopath, imageFolder, savePath):

        videos = sorted(glob.glob(videopath+"/*.mp4"))
        if(len(videos)>1): #stitch videos
            self.videopath = self.concateVideo(videopath)
        else:
            self.videopath = videos[0]

        self.imageFolder = imageFolder
        
        self.photospath = self.sortpath(imageFolder)
        self.count = len(self.photospath)
        self.savePath = savePath
        dir = os.path.dirname(savePath)
        self.chosenPath = os.path.join(dir, "chosen")
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        if not os.path.exists(self.chosenPath):
            os.makedirs(self.chosenPath)
        self.choose_id = -1
        self.queue = queue.Queue()


    """
        Sorts the image files in the specified folder based on the creation time stored in the EXIF metadata.

        Args:
            img_folder (str): The path to the folder containing the image files.

        Returns:
            numpy.ndarray: An array of strings representing the sorted filenames.
    """    
    def sortpath(self,img_folder):
        kv = {}
        # 遍历图像文件夹中的所有文件
        for filename in os.listdir(img_folder):
            if filename.endswith(".jpg") or filename.endswith(".JPG"):
                imgPath = os.path.join(img_folder, filename)
                exif_dict = piexif.load(imgPath)
                if 306 in exif_dict['0th'].keys():
                    ctime = exif_dict['0th'][306]
                    stime = str(ctime)[2:-1]
                    s_t = time.strptime(stime, "%Y:%m:%d %H:%M:%S")  # 返回元祖
                    mkt = int(time.mktime(s_t))
                    kv[imgPath] = mkt

        new_kv = sorted(kv.items(), key=lambda d: d[1], reverse=False)
        new_kvn = np.asarray(new_kv)
        sorted_filenames = new_kvn[:, 0]
        return sorted_filenames

    def findPeak(self, arr, n) : 
        # first or last element is peak element 
        if (n == 1) : 
            return 0
        if (arr[0] >= arr[1]) : 
            return 0
        if (arr[n - 1] >= arr[n - 2]) : 
            return n - 1
    
        # check for every other element 
        for i in range(1, n - 1) : 
            # check if the neighbors are smaller 
            if (arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]) : 
                return i 

    def concateVideo(self,path):
        save_path = os.path.join(path,"all.mp4")
        if os.path.exists(save_path):
            return save_path
        videos = sorted(glob.glob(path+"/*.mp4"))
        clips = []
        for video in videos:
            tmp = VideoFileClip(video)
            clips.append(tmp)
        final_clip = concatenate_videoclips(clips)
        
        final_clip.to_videofile(save_path,fps=clips[0].fps, remove_temp=False)
        return save_path
    def compareWithpHash(self, image, frame):
        h1 = pHash(image)
        h2 = pHash(frame)

        return cmpHash(h1, h2)

    def bySim(self, t):
        return -t[1]

    def getDatetime(self, file):
        exif_dict = piexif.load(file)
        # print(exif_dict)
        create_time = exif_dict['Exif'][36867]
        myBytes = create_time.decode()
        date1 = datetime.strptime(myBytes, "%Y:%m:%d %H:%M:%S")
        return date1


    def findMatch(self,cap,start,end,image,path, alpha=0.8):
        skip_count=0
        found = False
        self.frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frameID=start
        for i in range(start,end):
            frameState, frame = cap.read()
            frameID += 1
            if (frameState == False):
                continue
            else:
                his2 = cv2.calcHist(frame, [0], None, [256], [0, 256])
                if (np.count_nonzero(his2) < 10):  # skip bad frames
                    skip_count += 1
                else:
                    frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
                    # BGRtoRGB
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sim = self.compareWithpHash(image, frame)
                    if (sim >= alpha):
                        self.frames.append([frame, sim, frameID - 1])

                        # print(frameID)
        if (len(self.frames)>0):
                L = sorted(self.frames, key=self.bySim)
                self.choose_id = L[0][2]
                self.indexes.append(L[0][2])
                chosensave = os.path.join(self.chosenPath,str(L[0][2])+".jpg")
                cv2.imwrite(chosensave, L[0][0])
                print(path, self.choose_id,L[0][1])
                found=True
                # self.log.logger.info(path, self.choose_id,L[0][1])
        return found,skip_count

    def processVideo(self):
        cap = cv2.VideoCapture(self.videopath)
        isOpened = cap.isOpened

        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        step = math.floor(2 * fps)
        # frame id extract
        totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.indexes = []
        self.frames = []

        # image = cv2.imread(self.photospath[0])
        image = cv2.imdecode(np.fromfile(self.photospath[0],dtype=np.uint8),cv2.IMREAD_COLOR)
        self.width = image.shape[1]
        self.height = image.shape[0]
        date1 = self.getDatetime(self.photospath[0])

        frameID = 0
        idx = 0
        found = False
        skip = 0
        skip_count = 0
        num_images = len(self.photospath)
        pbar = tqdm(total=num_images)
        while (idx < num_images):
            curdate = self.getDatetime(self.photospath[idx])
            if (idx == 0):
                deltatime = 180 * fps
                start = 0
                end = deltatime
            else:
                deltatime = (curdate - date1).seconds
                halfw = int(deltatime/2)
                if (found == True):
                    image = cv2.imdecode(np.fromfile(self.photospath[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    start = self.choose_id + deltatime * fps - halfw * fps
                    end = self.choose_id + deltatime * fps + halfw * fps
                    # start = self.choose_id + 1
                    # end = self.choose_id + 180* fps
                    skip_count = 0
                else:
                    deltatime = (curdate - date1).seconds
                    start = self.choose_id + 1
                    # end = totalframes
                    end = min(self.choose_id + 2*deltatime * fps+skip_count, totalframes)
                    print(start, end)
            # compute hash value for [start,end] in cap
            # compute similarity value between target and cap


            curFind,skip_count_delta = self.findMatch(cap,start,end,image,self.photospath[idx])

            if(curFind==True):
                date1 = curdate
                skip_count += skip_count_delta
     
                found = True
                # print(self.photospath[idx])
            else:
                # if(len(self.indexes)!=0):
                #     news = start+1
                # else:
                #     news=0
                newe = min(start+2*(end-start),totalframes)
                print(start,newe)
                newcurFind,skip_count_delta = self.findMatch(cap, start, newe , image,self.photospath[idx],0.6)
                if(newcurFind==False):
                    found = False
                    print(self.photospath[idx], "not found")
                    self.indexes.append(-1)
                else:
                    # if(len(self.indexes)>1 and (self.indexes[-1]-self.indexes[-2]>=fps*30 or self.indexes[-1]-self.indexes[-2]<5)):
                    #     self.indexes[-1]
                    #     self.choose_id = self.indexes[-2]
                    # else:
                        date1 = curdate
                        skip_count += skip_count_delta
                        found = True
            
            idx += 1
            pbar.update(1)
        pbar.close()

        print(totalframes)
        extractframes = []
        i = self.indexes[0]
        j = 1
        while (i + step < totalframes):
            i = step + i
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            frameState, frame = cap.read()
            his2 = cv2.calcHist(frame, [0], None, [256], [0, 256])
            if (np.count_nonzero(his2) > 10):
                extractframes.append(i)
        np_frames = np.asarray(extractframes)

        # print(self.indexes)
        # for id in self.indexes:
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, id)
        #     frameState,frame = cap.read()
        #     frame = cv2.resize(frame, (self.width,self.height))
        #     # path = os.path.join(self.savePath,"match"+str(id)+".jpg")
        #     path = os.path.join(choosepath, str(id) + ".jpg")
        #     cv2.imwrite(path, frame)


        print("inserting GPS ...")
        for n, path in enumerate(self.indexes):
            if (n == len(self.indexes) - 1):
                break
            else:
                x_min = self.indexes[n]
                x_max = self.indexes[n + 1]
                if(x_min==-1 or x_max==-1):
                    print("no frames between %s and %s" % (self.photospath[n], self.photospath[n + 1]))
                    continue
                # print(x_min,x_max)
                pos = np.where((np_frames > x_min) & (np_frames < x_max))
                # insert gps
                # framenum = len(pos)
                if (len(pos[0])):
                    self.processGPS(self.photospath[n], self.photospath[n + 1], pos, cap, np_frames)
                else:
                    print("no frames between %d and %d" % (x_min, x_max))

        cap.release()

    def to_latlng(self, latlng):

        degree = int(latlng)
        res_degree = latlng - degree
        minute = int(res_degree * 60)
        res_minute = res_degree * 60 - minute
        seconds = round(res_minute * 60.0, 3)
        result = str(degree)+'/1 '+str(minute)+'/1 '+str(int(seconds * 1000))+'/1000'
        # return ((degree, 1), (minute, 1), (int(seconds * 1000), 1000))
        return result

    def read_modify_exif(self, image_path, frame, latitude, longitude, altitude, ref):
        """ read and modify exif"""
        # exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {},}
        # exif_bytes = piexif.dump(exif_dict)
        frame.save(image_path, "jpeg")
        ref_data = pyexiv2.Image(ref)
        cur_data = pyexiv2.Image(image_path)
        pyexiv2.registerNs('a namespace for interpolation', 'drone-dji')
        xmp_dixt = ref_data.read_xmp()
        r = xmp_dixt['Xmp.drone-dji.GimbalRollDegree']
        y = xmp_dixt['Xmp.drone-dji.GimbalYawDegree']
        p = xmp_dixt['Xmp.drone-dji.GimbalPitchDegree']
        exif_dict = ref_data.read_exif()
        longref = exif_dict['Exif.GPSInfo.GPSLongitudeRef']
        latref = exif_dict['Exif.GPSInfo.GPSLatitudeRef']

        lat = self.to_latlng(latitude)
        long = self.to_latlng(longitude)
        alt = str(int(altitude * 1000)) + '/1000'
        new_xmp_dict = {'Xmp.drone-dji.GpsLatitude': str(latitude), 'Xmp.drone-dji.GpsLongitude': str(longitude),
                        'Xmp.drone-dji.AbsoluteAltitude': str(altitude), 'Xmp.drone-dji.GimbalRollDegree': r,
                        'Xmp.drone-dji.GimbalYawDegree': p, 'Xmp.drone-dji.GimbalPitchDegree': y}
        new_exif_dict = {'Exif.GPSInfo.GPSLatitude': lat, 'Exif.GPSInfo.GPSLongitude': long,
                          'Exif.GPSInfo.GPSAltitude': alt,
                          'Exif.GPSInfo.GPSLatitudeRef':longref,'Exif.GPSInfo.GPSLongitudeRef':latref}
        cur_data.modify_xmp(new_xmp_dict)
        cur_data.modify_exif(new_exif_dict)

        ref_data.close()
        cur_data.close()
    

    def get_GPS_exif(image_path):
        exif_dict = piexif.load(image_path)
        long = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
        lat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
        alt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]

    def processGPS(self, path1, path2, pos, cap, frames):
        imageA = piexif.load(path1)
        imageA_gps_lat = imageA['GPS'][2][0][0] / imageA['GPS'][2][0][1] + \
                         imageA['GPS'][2][1][0] / imageA['GPS'][2][1][1] / 60 + imageA['GPS'][2][2][0] / \
                         imageA['GPS'][2][2][1] / 3600
        imageA_gps_long = imageA['GPS'][4][0][0] / imageA['GPS'][4][0][1] + imageA['GPS'][4][1][0] / \
                          imageA['GPS'][4][1][1] / 60 + imageA['GPS'][4][2][0] / imageA['GPS'][4][2][1] / 3600
        imageA_gps_alt = imageA['GPS'][6][0] / imageA['GPS'][6][1]

        imageB = piexif.load(path2)
        imageB_gps_lat = imageB['GPS'][2][0][0] / imageB['GPS'][2][0][1] + imageB['GPS'][2][1][0] / \
                         imageB['GPS'][2][1][1] / 60 + imageB['GPS'][2][2][0] / imageB['GPS'][2][2][1] / 3600
        imageB_gps_long = imageB['GPS'][4][0][0] / imageB['GPS'][4][0][1] + imageB['GPS'][4][1][0] / \
                          imageB['GPS'][4][1][1] / 60 + imageB['GPS'][4][2][0] / imageB['GPS'][4][2][1] / 3600
        imageB_gps_alt = imageB['GPS'][6][0] / imageB['GPS'][6][1]

        framenum = len(pos[0])
        diff_long = (imageB_gps_long - imageA_gps_long) / framenum
        diff_lat = (imageB_gps_lat - imageA_gps_lat) / framenum
        diff_alt = (imageB_gps_alt - imageA_gps_alt) / framenum

        for i, id in enumerate(pos[0]):
            # print(frames[id])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[id])
            frameState, frame = cap.read()
            frame = cv2.resize(frame, (self.width, self.height))
            # 转变成Image
            frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 格式转变，BGRtoRGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frameState):
                img_name = str(frames[id]) + '.jpg'
                img_path = os.path.join(self.savePath, img_name)

                # save_path = os.path.join(pic_dir, 'gps_' + img_name)
                img_latitude = imageA_gps_lat + diff_lat * (i + 1)
                img_longitude = imageA_gps_long + diff_long * (i + 1)
                img_altitude = imageA_gps_alt + diff_alt * (i + 1)
                # print(type(frame))
                self.read_modify_exif(img_path, frame, img_latitude, img_longitude, img_altitude,path1)

    def run(self):
        self.processVideo()
        # self.processGPS()


# 均值哈希算法
def aHash(img, shape=(10, 10)):
    # 缩放为10*10
    img = cv2.resize(img, shape)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 100
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值感知算法
def dHash(img, shape=(10, 10)):
    # 缩放10*11
    img = cv2.resize(img, (shape[0] + 1, shape[1]))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 感知哈希算法(pHash)
def pHash(img, shape=(10, 10)):
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:10, 0:10]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 通过得到RGB每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# 计算单通道的直方图的相似值
"""
    Calculate the degree of histogram overlap between two images.

    Args:
        image1 (numpy.ndarray): The first image.
        image2 (numpy.ndarray): The second image.

    Returns:
        float: The degree of histogram overlap.

"""
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# Hash值对比
def cmpHash(hash1, hash2, shape=(10, 10)):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 相等则n计数+1，n最终为相似度
        if hash1[i] == hash2[i]:
            n = n + 1
    return n / (shape[0] * shape[1])


if __name__ == "__main__":
    video = VideoProcess('/home/a62/GMP/open3d/jiangxi/B05/videos', '/home/a62/GMP/open3d/jiangxi/B05/images', \
                         '/home/a62/GMP/open3d/jiangxi/B05/frames')
    time1 = time.perf_counter()
    video.run()
    time2 = time.perf_counter()
    print(time2 - time1)

    # img1 = cv2.imread('/home/a119/Documents/GMP/cpp/videoProcess/data/images/【全覆盖】0002_2023-08-08-09-32-04.JPG')
    # img2 = cv2.imread('/home/a119/Documents/GMP/cpp/videoProcess/frames/22.jpg')
    # # img_diff = Image.new("RGBA", img.size)
    # # time1 = time.perf_counter()
    # # mismatch = pixelmatch(img1, img2, img_diff, includeAA=True)
    # # time2 = time.perf_counter()
    # # print((time2-time1))
    # # print(mismatch)

    # hash1 = aHash(img1)
    # hash2 = aHash(img2)
    # n = cmpHash(hash1, hash2)
    # print('均值哈希算法相似度：', n)

    # hash1 = dHash(img1)
    # hash2 = dHash(img2)
    # n = cmpHash(hash1, hash2)
    # print('差值哈希算法相似度：', n)

    # hash1 = pHash(img1)
    # hash2 = pHash(img2)
    # n = cmpHash(hash1, hash2)
    # print('感知哈希算法相似度：', n)

    # n = classify_hist_with_split(img1, img2)
    # print('三直方图算法相似度：', n[0])

    # n = calculate(img1, img2)
    # print('单通道的直方图算法相似度：', n[0])

