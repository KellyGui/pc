import glob
import math
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import os
import time
from PIL import Image
import piexif
import numpy as np
from moviepy.editor import *
import pyexiv2

# TODO
'''
1. 分别给两个文件夹的图排序
2. 原图去配抽图，
    2.2并找到原图在抽图的哪两个图中间
3. 根据最终的位置顺序图，给抽图计算坐标（每个图之间的模，按比例计算位置）
'''

class video2pic:

    def __init__(self, videopath, imagesfiles,savePath):
        # self.videopath = '/data/file/share/data/08/video/'
        # self.imagesfiles = '/data/file/share/data/08/images/' # 原图
        # self.picfile = '/data/file/share/data/08/pics/'
        self.videopath = videopath
        self.imagesfiles = imagesfiles # 原图
        self.projfile = os.path.abspath(os.path.dirname(os.path.dirname(videopath)))
        self.print_log = open(self.projfile+"/printlog.txt","w")
        sys.stdout = self.print_log
        '''
        0.3 删除红外图
        '''
        print('0.3 删除红外图')
        
        filelist=os.listdir(self.imagesfiles) 
        for file in filelist:
            if 'INFRA' in file:
                del_file = self.imagesfiles + file #当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
                os.remove(del_file)#删除文件
                print("已经删除：",del_file)
        

        self.picfile = savePath
        print(self.picfile)
        if not os.path.exists(self.picfile):
            os.makedirs(self.picfile)
        else :
            self.delete_files(self.picfile)
        self.framesdata_len = 0
        self.frameskeydes = []
        self.imageskeydes = []#原图的
        self.bf = cv2.BFMatcher()
        self.sift = cv2.SIFT_create()
        self.img_folder = self.imagesfiles
        kv = {}

        # 遍历图像文件夹中的所有文件
        for filename in os.listdir(self.img_folder):
            if filename.endswith(".jpg") or filename.endswith(".JPG"):
                imgPath = self.img_folder + filename
                exif_dict = piexif.load(imgPath)
                if 306 in exif_dict['0th'].keys():
                    ctime = exif_dict['0th'][306]
                    stime = str(ctime)[2:-1]
                    s_t = time.strptime(stime, "%Y:%m:%d %H:%M:%S")  # 返回元祖
                    mkt = int(time.mktime(s_t))
                    kv[imgPath] = mkt

        new_kv = sorted(kv.items(),  key=lambda d: d[1], reverse=False)
        print(new_kv)
        # [('/data/file/share/data/龙游塔石/04/images/уАРхЕишжЖчЫЦ2._VIS_0002_2023-10-26-16-08-07.JPG', 1698115665), ('/data/file/share/data/龙游塔石/04/images/уАРхЕишжЖчЫЦ2._VIS_0004_2023-10-26-16-08-07.JPG', 1698115677), ('

        new_kvn = np.asarray(new_kv)
        self.sorted_largeiamge_path = new_kvn[:, 0]
        print(self.sorted_largeiamge_path)
        # ['/data/file/share/data/龙游塔石/04/images/уАРхЕишжЖчЫЦ2._VIS_0002_2023-10-26-16-08-07.JPG' '/data/file/share/data/龙游塔石/04/images/уАРхЕишжЖчЫЦ2._VIS_0004_2023-10-26-16-08-07.JPG'
        self.ref_data = pyexiv2.Image(self.sorted_largeiamge_path[0])
    '''
    函数整理
    '''
    def delete_files(self,directory):
        file_list = os.listdir(directory)
        for file in file_list:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    def two_image_posision(self,kd2,kd1):
        relative_position = None
        keypoints2, descriptors2 = kd2
        keypoints1,descriptors1 = kd1
        if descriptors2 is None:
            return None
        # 匹配特征点
        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
        # 应用比率测试
        good_matches = []
        if len(matches) > 4 and len(matches[0]) >= 2:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            if len(good_matches) > 4:
                # 计算相对位置
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                # print(src_pts,dst_pts)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    relative_position = [M[0][2],M[1][2]]
        return relative_position

    def image_posision(self,sourceindex,kd1):# 找到大图确切位置，需要考虑前张或后张为信号丢失图
        if sourceindex >= self.framesdata_len:
            return  sourceindex - 1
        relative_position21 =  self.two_image_posision(self.frameskeydes[sourceindex-1],kd1)
        relative_position22 =  self.two_image_posision(self.frameskeydes[sourceindex],kd1)
        relative_position23 =  self.two_image_posision(self.frameskeydes[sourceindex+1],kd1)
        if relative_position22 is None:
            return  sourceindex - 1
        if relative_position21 is not None:
            len1 = np.linalg.norm(x=relative_position21, ord=2) + np.linalg.norm(x=relative_position22, ord=2)
        else:
            return  sourceindex

        if relative_position23 is not None:
            len2 = np.linalg.norm(x=relative_position23, ord=2) + np.linalg.norm(x=relative_position22, ord=2)
        else:
            return  sourceindex - 1
        
        if len1<len2:
            return  sourceindex - 1
        return sourceindex

    def to_latlng(self,latlng):
        degree = int(latlng)
        res_degree = latlng - degree
        minute = int(res_degree * 60)
        res_minute = res_degree * 60 - minute
        seconds = round(res_minute * 60.0, 3)
        result = str(degree)+'/1 '+str(minute)+'/1 '+str(int(seconds * 1000))+'/1000'
        return result
    def read_modify_exif(self,image_path, latitude, longitude, altitude):
        """ read and modify exif"""
        cur_data = pyexiv2.Image(image_path)
        pyexiv2.registerNs('a namespace for interpolation', 'drone-dji')
        xmp_dixt = self.ref_data.read_xmp()
        r = xmp_dixt['Xmp.drone-dji.GimbalRollDegree']
        y = xmp_dixt['Xmp.drone-dji.GimbalYawDegree']
        p = xmp_dixt['Xmp.drone-dji.GimbalPitchDegree']
        exif_dict = self.ref_data.read_exif()
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
        cur_data.close()

    def format_latlng(self,latlng):
        """经纬度十进制转为分秒"""
        degree = int(latlng)
        res_degree = latlng - degree
        minute = int(res_degree * 60)
        res_minute = res_degree * 60 - minute
        seconds = round(res_minute * 60.0,3)
        return ((degree, 1), (minute,1), (int(seconds*1000), 1000))

    def format_latlng_b(self,latlng):
        """经纬度分秒转为十进制"""
        degree = latlng[0][0]
        minute = latlng[1][0]
        seconds = latlng[2][0]/latlng[2][1]
        return degree + minute/60.0 + seconds/3600.0

    def run(self):
        '''
        0.2 视频抽图到内存
        '''
        print('0.2 视频抽图到内存')
        framesdata = []
        videomp4paths = sorted(glob.glob(self.videopath+"*.mp4"))
        print(videomp4paths)
        for videomp4path in videomp4paths:
            cap = cv2.VideoCapture(videomp4path)
            isOpened = cap.isOpened

            fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
            step = math.floor(2 * fps)# 秒
            totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            sum = 0
            imageNum = 0
            while (isOpened):
                sum += 1
                (frameState, frame) = cap.read()  # 记录每帧及获取状态
                if sum <= totalframes  and frameState == True and (sum % step==0):
                    his2 = cv2.calcHist(frame, [0], None, [256], [0, 256])
                    his2nn = np.count_nonzero(his2)
                    if his2nn < 10:
                        print("skip")
                        continue
                    # 格式转变，BGRtoRGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame,(1440,1080))
                    # 转变成Image
                    frame = Image.fromarray(np.uint8(frame))
                    frame = np.array(frame)
                    # RGBtoBGR满足opencv显示格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                    framesdata.append(frame)
                    imageNum = imageNum + 1
                elif frameState == False:
                    break
                if sum > totalframes:
                    break
            cap.release()

        '''
        2.1 原图去匹配
        '''
        print('2.1 原图去匹配')
        self.framesdata_len = len(framesdata)

        for i in range(self.framesdata_len):
            image2 = framesdata[i]
            image2 = cv2.resize(image2,(200,math.floor(200*image2.shape[1]/image2.shape[0])))
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            keypoints2, descriptors2 = self.sift.detectAndCompute(gray2, None)
            self.frameskeydes.append([keypoints2, descriptors2])
        largeposition = []
        last_index = 0
        index =0
        largeindex = 0
        sims_ad = []
        for largeimage in self.sorted_largeiamge_path:
            sims = []
            image1 = cv2.imread(largeimage)
            image1 = cv2.resize(image1,(200,math.floor(200*image1.shape[1]/image1.shape[0])))# TODO 压缩大小
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            keypoints1, descriptors1 = self.sift.detectAndCompute(gray1, None)
            self.imageskeydes.append([keypoints1, descriptors1])
            last_indexnums = self.framesdata_len-index
            simindex = 0
            matched = 0
            for frameindextmp in range(last_indexnums):
                keypoints2, descriptors2 = self.frameskeydes[frameindextmp+last_index]
                if descriptors2 is None or descriptors2.size < 200:# or descriptors2.size < 25000
                    sims.append(0)
                else:
                    # 匹配特征点
                    matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
                    # 应用比率测试
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                    good_matches_len = len(good_matches)
                    sims.append(good_matches_len)

                    if good_matches_len>16 :#TODO 匹配点参数 500 | resize200 30
                        matched = 1
                    if simindex >= 10 and matched:# 窗型计算
                        simsmax = np.argmax(sims)
                        if simsmax != simindex:
                            locat = self.image_posision(simsmax + last_index,[keypoints1, descriptors1])
                            largeposition.append([largeindex,locat,simsmax + last_index,largeimage])# 原图索引、再哪帧后面、和哪帧最匹配、原图地址
                            sims_ad.append(simsmax)
                            print(simsmax + last_index,sims[simsmax],largeimage,largeindex)
                            print(sims)
                            last_index = simsmax + last_index
                            index = last_index
                            break
                index +=1
                simindex +=1
                if index >= self.framesdata_len:# 避免原图找不到配图（已经去除了丢信号的图情况）
                    index = last_index
                    print(sims)
                    break
            if index >= self.framesdata_len:
                break
            largeindex +=1
        print(largeposition)

        '''
        3.1 保存抽图并写入GPS
        '''
        print("3.1 保存抽图并写入GPS")
        findnums = len(largeposition)
        for imagesindex in range(findnums-1):
            largeindex1 = largeposition[imagesindex][0]# 原图索引、在哪帧后面、和哪帧最匹配、原图地址
            largeindex2 = largeposition[imagesindex+1][0]
            largeindex_e = largeindex2 - largeindex1
            if largeindex_e != 1: # 如果下一张和这张不连续，则这张之后的几帧无法计算
                continue
            locat1 = largeposition[imagesindex][1]
            locat2 = largeposition[imagesindex+1][1]
            locat_e = locat2 - locat1
            # 计算GPS向量及模
            imageA = piexif.load(largeposition[imagesindex][3])
            imageA_gps_lat = imageA['GPS'][2][0][0] / imageA['GPS'][2][0][1] + \
                                imageA['GPS'][2][1][0] / imageA['GPS'][2][1][1] / 60 + imageA['GPS'][2][2][0] / \
                                imageA['GPS'][2][2][1] / 3600
            imageA_gps_long = imageA['GPS'][4][0][0] / imageA['GPS'][4][0][1] + imageA['GPS'][4][1][0] / \
                                imageA['GPS'][4][1][1] / 60 + imageA['GPS'][4][2][0] / imageA['GPS'][4][2][1] / 3600
            imageA_gps_alt = imageA['GPS'][6][0] / imageA['GPS'][6][1]

            imageB = piexif.load(largeposition[imagesindex+1][3])
            imageB_gps_lat = imageB['GPS'][2][0][0] / imageB['GPS'][2][0][1] + imageB['GPS'][2][1][0] / \
                                imageB['GPS'][2][1][1] / 60 + imageB['GPS'][2][2][0] / imageB['GPS'][2][2][1] / 3600
            imageB_gps_long = imageB['GPS'][4][0][0] / imageB['GPS'][4][0][1] + imageB['GPS'][4][1][0] / \
                                imageB['GPS'][4][1][1] / 60 + imageB['GPS'][4][2][0] / imageB['GPS'][4][2][1] / 3600
            imageB_gps_alt = imageB['GPS'][6][0] / imageB['GPS'][6][1]

            diff_long = imageB_gps_long - imageA_gps_long#经度
            diff_lat = imageB_gps_lat - imageA_gps_lat#维度
            diff_alt = imageB_gps_alt - imageA_gps_alt#高度

            gpsnorm = np.linalg.norm(x=[diff_long,diff_lat], ord=2)
            print(diff_long,diff_lat)
            # 计算所有向量及模
            framesv = []
            framesvex = [0,0]
            framesv_len = []

            for framesv_index in range(locat_e):
                if framesv_index == 0:
                    relative_position_f = self.two_image_posision(self.frameskeydes[locat1+1],self.imageskeydes[largeindex1])
                    print(locat1+1,".jpg",relative_position_f)
                else:
                    la = locat1+1+framesv_index
                    lb = locat1+framesv_index
                    relative_position_f = self.two_image_posision(self.frameskeydes[la],self.frameskeydes[lb])
                    print(la,".jpg",relative_position_f)
                if relative_position_f is None:
                    break
                vx , vy = relative_position_f[0],relative_position_f[1]
                framesvex = [framesvex[0] + vx , framesvex[1]+vy]
                framesv.append([vx,vy])
                vlen = np.linalg.norm(x=[vx,vy], ord=2)
                framesv_len.append(vlen)
            if relative_position_f is None:
                continue
            relative_position_f = self.two_image_posision(self.imageskeydes[largeindex1+1],self.frameskeydes[locat2])
            print(locat2,"+.jpg",relative_position_f)
            if relative_position_f is None:
                continue
            vx , vy = relative_position_f[0],relative_position_f[1]
            framesvex = [framesvex[0] + vx , framesvex[1]+vy]
            framesv.append([vx,vy])
            vlen = np.linalg.norm(x=[vx,vy], ord=2)
            framesv_len.append(vlen)
            framesv_lens = np.linalg.norm(x=framesvex, ord=2)

            # # 处理特殊的点
            # if (max(framesv_len) / framesv_lens) > 0.5:
            #     framesv_lenmax_i = np.argmax(framesv_len)
            #     framesvex = [framesvex[0] - framesv[framesv_lenmax_i][0] , framesvex[1]-framesv[framesv_lenmax_i][1]]
            #     framesv[framesv_lenmax_i] = [0,0]
            #     framesv_lens = np.linalg.norm(x=framesvex, ord=2)
            print("gpsnorm",gpsnorm," framesv_lens",framesv_lens,largeposition[imagesindex][3])
            print("framesv",framesv)
            print("framesv_len",framesv_len)
            print("framesvex",framesvex)

            # TODO 排除模过长的情况，暂时没考虑

            # 推算每张图的GPS位置并保存图片
            dlong = 0
            dlat = 0
            dalt = 0
            k = gpsnorm / framesv_lens
            l_x=gpsnorm
            l_y=framesv_lens
            vex_d = diff_long/l_x*l_y - framesvex[0], diff_lat/l_x*l_y - framesvex[1]

            print('向量的模=',l_x,l_y)
            print("目标向量的基",diff_long/l_x,diff_lat/l_x)
            print("变化后向量",diff_long/l_x*l_y,diff_lat/l_x*l_y)
            print("向量差",vex_d)
            framesv_len = np.array(framesv_len)
            framesv_len_sum = np.sum(framesv_len)
            for framesv_index in range(locat_e):
                # bi = framesv_len[framesv_index]/framesv_lens
                bi = framesv_len[framesv_index]/framesv_len_sum
                # dlong += framesv[framesv_index][0]*k #这种计算会产生小斜线
                # dlat += framesv[framesv_index][1]*k

                # dlong += diff_long*bi#这种计算是两个原图间的直线
                # dlat += diff_lat*bi

                dlong += (framesv[framesv_index][0] + vex_d[0]*bi)*k #
                dlat += (framesv[framesv_index][1] + vex_d[1]*bi)*k

                dalt += diff_alt*bi
                frameindex_ = locat1+1+framesv_index
                frameindex_long = imageA_gps_long + dlong # 经度
                frameindex_lat = imageA_gps_lat + dlat # 维度
                frameindex_alt = imageA_gps_alt + dalt
                fileName = self.picfile + str(frameindex_) + '.jpg'  # 存储路径
                cv2.imwrite(fileName, framesdata[frameindex_], [cv2.IMWRITE_JPEG_QUALITY, 100])
                self.read_modify_exif(fileName, frameindex_lat, frameindex_long, frameindex_alt)

        self.ref_data.close()
        
        print("finishiTime:",time.time())

        image_paths = os.listdir(self.picfile)
        image_paths.sort(key=lambda x:int(x[:-4])) #对读取的路径进行排序

        alist = []
        for filename in image_paths:
            if filename.endswith(".jpg") or filename.endswith(".JPG"):
                exif_dict = piexif.load(self.picfile + filename)
                high = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]  # = (int(100*round(altitude,2)),100)  # 修改高度，GPSAltitude是内置变量，不可修改
                lon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]  # = format_latlng(longitude)  # 修改经度
                lat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]  # = format_latlng(latitude)  # 修改纬度

                lon = self.format_latlng_b(lon)
                lat = self.format_latlng_b(lat)
                high = high[0]/high[1]
                alist.append([filename, lat, lon, high])

        def drawPic(l):
            plt.figure(figsize=(10,10))
            plt.axis("equal")
            plt.xlabel('Longitude')    #x轴的标题
            plt.ylabel('Latitude')    #y轴的标题
            color1 = '#00FF00'
            color2 = '#0000FF'
            color = '#00FF00'

            #连接各个点
            for i in range(len(l)-1):
                start = (l[i][2],l[i+1][2])
                end = (l[i][1],l[i+1][1])
                dx = l[i][2] - l[i+1][2]
                dy = l[i][1] -l[i+1][1]
                lens = math.sqrt(dx*dx + dy*dy)
                if lens> 200:
                    plt.plot(start,end,color='#FF0000')
                    if color == color1:
                        color = color2
                    else:
                        color = color1
                else:
                    plt.plot(start,end,color='#0085c3')
                if (i % 20) == 0:
                    plt.text(l[i+1][2],l[i+1][1],l[i+1][0],color= color)
                plt.plot(l[i+1][2],l[i+1][1],'.',color= color)
            plt.show()
            plt.savefig(self.projfile + '/picslocate.png')
        drawPic(alist)
        self.print_log.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":

    # video = video2pic("/data/file/share/data/龙游塔石/00/videos/", "/data/file/share/data/龙游塔石/00/images/", "/data/file/share/data/龙游塔石/00/pics/")
    # video.run()
    video = video2pic("/home/a62/GMP/open3d/jiagnshanshimen/10/videos/", "/home/a62/GMP/open3d/jiagnshanshimen/10/images/", "/home/a62/GMP/open3d/jiagnshanshimen/10/pics/")
    # video.run()
    # video = video2pic("/data/file/share/data/万田/08/videos/", "/data/file/share/data/万田/08/images/", "/data/file/share/data/万田/08/pics/")
    # video.run()
    # video = video2pic("/data/file/share/data/万田/09/videos/", "/data/file/share/data/万田/09/images/", "/data/file/share/data/万田/09/pics/")
    time1 = time.perf_counter()
    video.run()
    time2 = time.perf_counter()
    print(time2 - time1)
