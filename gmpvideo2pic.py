# coding:utf-8
import glob
import os

import PIL
import cv2
from PIL import Image
import numpy as np
import piexif
from tqdm import tqdm
import time
from datetime import datetime
import math
import queue
import logging
# from logger import Logger
from concatvideo import concateVideo
from pixelmatch.contrib.PIL import pixelmatch
import pyexiv2
# from extractFrames import compare_snnr,SURF

# choosepath = r"D:\GMP\videoReconstruction\odmResult\longyou08\06\chosen"
# logpath=r"D:\GMP\videoReconstruction\DatagenerateLog\test"

class VideoProcess:
    def __init__(self, videopath, imageFolder, savePath):

        videos = sorted(glob.glob(videopath+"/*.mp4"))
        if(len(videos)>1): #stitch videos
            self.videopath = concateVideo(videopath)
        else:
            self.videopath = videos[0]

        self.imageFolder = imageFolder
        # self.photospath = sorted(glob.glob(imageFolder + r'\*.JPG'))
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
        pyexiv2.registerNs('a namespace for interpolation', 'drone-dji')

    def sortpath(self,img_folder):
        kv = {}
        # 遍历图像文件夹中的所有文件
        for filename in os.listdir(img_folder):
            if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".jpeg") :
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


    def findMatch(self,cap,start,end,image,path,alpha=0.8,flag=0):
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
                        self.frames.append([frame,sim, frameID - 1])

                        # print(frameID)
        if (len(self.frames)>0):
                #for topk, compute pixel error
                L = sorted(self.frames, key=self.bySim)
                tmp_choose=0
                # if flag != 0:
                #     k = len(L)
                #     minerror = 0
                #     tmp_choose =-1
                #     kp1, des1 = SURF(image)
                #     for i in range(0,k):
                #         cap.set(cv2.CAP_PROP_POS_FRAMES, L[i][1])
                #         frameState, frame = cap.read()
                #         frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
                #         good = compare_snnr(frame,kp1,des1,image)
                #         if minerror< good:
                #             minerror = good
                #             tmp_choose = i
                #             chosen_img = frame
                # else:
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, L[0][1])
                #     frameState, frame = cap.read()
                #     frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
                #     chosen_img = frame
                self.choose_id = L[tmp_choose][2]
                self.indexes.append(L[tmp_choose][2])
                chosensave = os.path.join(self.chosenPath,str(L[tmp_choose][2])+".jpg")
                cv2.imwrite(chosensave,L[tmp_choose][0])
                print(path, self.choose_id,L[tmp_choose][1])
                found=True
                # self.log.logger.info(path, self.choose_id,L[0][1])
        return found,skip_count

    def processVideo(self):
        cap = cv2.VideoCapture(self.videopath)
        isOpened = cap.isOpened

        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        step = math.floor(1.5 * fps)
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
                deltatime = 300 * fps
                start = 0
                end = deltatime
            else:
                deltatime = (curdate - date1).seconds
                halfw = int(deltatime )
                if (found == True):
                    image = cv2.imdecode(np.fromfile(self.photospath[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
                    # deltatime = (curdate - date1).seconds
                    start = self.choose_id + deltatime * fps - halfw * fps+1
                    end = self.choose_id + deltatime * fps + halfw * fps
                    skip_count = 0
                else:

                    start = self.choose_id + 1
                    # end = totalframes
                    end = min(self.choose_id + 2*deltatime * fps + fps + skip_count, totalframes)
                    print(start, end)
            # compute hash value for [start,end] in cap
            # compute similarity value between target and cap


            curFind,skip_count_delta = self.findMatch(cap,start,end,image,self.photospath[idx])
            bonus =1
            # skip_count += skip_count_delta
            # while(curFind==False and bonus<10):
            #     bonus+=1
            #     end =  min(self.choose_id + deltatime *np.power(4,bonus)* fps + fps + skip_count, totalframes)
            #     curFind,skip_count_delta = self.findMatch(cap, start, end, image)
            #     skip_count += skip_count_delta

            if(curFind==True):
                date1 = curdate
                skip_count += skip_count_delta
                found = True
                # print(self.photospath[idx])
            else:
                # news = min(start,self.indexes[-1])
                # if(news<0):
                #     news=0
                newe = min(start + 2 * (end - start), totalframes)
                start = max(0,self.choose_id)
                print(start,end)
                newcurFind,skip_count_delta = self.findMatch(cap, start, end, image,self.photospath[idx],0.7)
                if(newcurFind==False):
                    found = False
                    print(self.photospath[idx], "not found")
                else:
                    date1 = curdate
                    skip_count += skip_count_delta
                    found = True
                    # print(self.photospath[idx])

            # self.frames = []
            # cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            # frameID = start
            # while (isOpened):
            #     (frameState, frame) = cap.read()
            #     frameID += 1
            #     # print(frameID)
            #     if (frameState == False):
            #         break
            #     else:
            #         his2 = cv2.calcHist(frame, [0], None, [256], [0, 256])
            #         if (np.count_nonzero(his2) < 10):  #skip bad frames
            #             skip = frameID - 1
            #             # frameID += 1
            #             skip_count += 1
            #             # pp = os.path.join("/home/a119/Documents/GMP/cpp/videoProcess/skip",str(skip)+'.jpg')
            #             # cv2.imwrite(pp,frame)
            #             # print("skip ",skip)
            #             continue
            #         else:
            #             frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
            #             # BGRtoRGB
            #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #             sim = self.compareWithpHash(image, frame)
            #             if (sim >= 0.8):
            #                 self.frames.append([frame, sim, frameID - 1])
            #                 # print(frameID)
            #             if (frameID - 1 >= end):
            #                 if (len(self.frames)):
            #                     L = sorted(self.frames, key=self.bySim)
            #                     self.choose_id = L[0][2]
            #                     self.indexes.append(L[0][2])
            #                     fullpath, name = os.path.split(self.photospath[idx])
            #                     print(name, L[0][2], L[0][1])
            #                     found = True
            #                     date1 = curdate
            #                     break
            #                 else:
            #                     found = False
            #                     # find from start to video end
            #
            #                     self.indexes.append(start)
            #                     print(self.photospath[idx], " not found")
            #                     print("start ",start)
            #                     break
            #     # frameID +=1
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

        print(self.indexes)
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
                if (x_min == -1 or x_max == -1):
                    print("no frames between %s and %s" % (self.photospath[n], self.photospath[n + 1]))
                    continue
                # print(x_min,x_max)
                pos = np.where((np_frames > x_min) & (np_frames < x_max))
                # insert gps
                # framenum = len(pos)
                if (len(pos[0])):
                    self.processGPS(self.photospath[n], self.photospath[n + 1], pos, cap, np_frames)

        cap.release()

    def to_latlng(self, latlng):

        degree = int(latlng)
        res_degree = latlng - degree
        minute = int(res_degree * 60)
        res_minute = res_degree * 60 - minute
        seconds = round(res_minute * 60.0, 3)
        result = str(degree) + '/1 ' + str(minute) + '/1 ' + str(int(seconds * 1000)) + '/1000'
        # return ((degree, 1), (minute, 1), (int(seconds * 1000), 1000))
        return result

    def read_modify_exif(self, image_path, frame, latitude, longitude, altitude,ref):
        """ read and modify exif"""
        frame.save(image_path, "jpeg")
        ref_data = pyexiv2.Image(ref)
        cur_data = pyexiv2.Image(image_path)

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
                         'Exif.GPSInfo.GPSLatitudeRef': longref, 'Exif.GPSInfo.GPSLongitudeRef':latref}
        cur_data.modify_xmp(new_xmp_dict)
        cur_data.modify_exif(new_exif_dict)

        ref_data.close()
        cur_data.close()
        # # exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {},}
        # # exif_bytes = piexif.dump(exif_dict)
        #
        # exif_dict = piexif.load(r"D:\GMP\download\yigu\22.jpg")
        # exif_bytes = piexif.dump(exif_dict)
        # exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(100 * round(altitude, 2)), 100)
        # exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = self.to_latlng(longitude)
        # exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = self.to_latlng(latitude)
        # # add GPS info
        # exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = 'E'
        # exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = 'N'
        # exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef] = 0
        # exif_dict['GPS'][piexif.GPSIFD.GPSMapDatum] = 'WGS-84'
        # exif_dict['GPS'][piexif.GPSIFD.GPSStatus] = 'A'
        #
        # exif_bytes = piexif.dump(exif_dict)
        # # piexif.insert(exif_bytes, image_path)
        # if (os.path.exists(image_path) == False):
        #     frame.save(image_path, "jpeg", exif=exif_bytes)

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
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                self.read_modify_exif(img_path, frame, img_latitude, img_longitude, img_altitude, path1)

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
    video = VideoProcess(r'/server-w/3Dreconstruction/杭州视频建模180/B01/videos', r'/server-w/3Dreconstruction/杭州视频建模180/B01/images', \
                         r'/server-w/3Dreconstruction/杭州视频建模180/B01/result')
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

