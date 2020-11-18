import cv2
import numpy as np
import os
from skimage.measure import compare_ssim
from scipy.misc import imread
from image_match.goldberg import ImageSignature
import time

from ImgDepthEstimation.ImgDepthEstimation import ImgDepthEstimation
from SalientObjectDetection.SalientObjectDetection import SalientObjectDetection

def textureRegionMatch(img1,img2,ymin,ymax,xmin,xmax):

    template = img1[ymin:ymax,xmin:xmax,:]
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    template = cv2.equalizeHist(template)
    h, w = template.shape
    gis = ImageSignature() 
    a = gis.generate_signature(template)

    # print(img2.shape)
    img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]), interpolation=cv2.INTER_AREA)  
    # print(img2.shape)
    target = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    height, width = target.shape 

    # print(target.shape)
    
    img_array = target[max(0,ymin-h):min(ymax+h,height),max(0,xmin-w):min(xmax+w,width)]
    img_array = cv2.equalizeHist(img_array)
    ht, wd = img_array.shape
    # print(img_array.shape)
    dis = []
    step = h//20
    i = 0
    # print(i*step+h)
    while (i*step+h)<ht:
        j = 0
        while (j*step+w)<wd:
            b = gis.generate_signature(img_array[i*step:i*step+h,j*step:j*step+w])
            dis.append(gis.normalized_distance(a, b))
            # print([i,j])
            j = j + 1
        i = i + 1

    index = dis.index(min(dis))
    y = index//j
    x = index%j

    # print(index)
    yimin = y*step + max(0,ymin-h)
    yimax = y*step+h + max(0,ymin-h)
    ximin = x*step + max(0,xmin-w)
    ximax = x*step+w + max(0,xmin-w)
    # print([ymin-ymax,xmin-xmax])
    crop = img2[yimin:yimax,ximin:ximax,:]
    return crop

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

# def textureMap(image_color):
#     img = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
#     height, width = img.shape

#     img = cv2.bilateralFilter(img, 0, 100, 10)
#     th = np.median(img.flatten())
#     edge = cv2.Canny(img,0,1*th)
#     edge = cv2.medianBlur(edge, 3)

#     return edge

def textureMap(image_color):
    img = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    img = cv2.bilateralFilter(img, 5, 100, 10)
    th = np.median(img.flatten())
    edge = cv2.Canny(img,0,1.5*th)
    # edge = cv2.medianBlur(edge, 3)

    return edge

def motionObjectRemove(image_color):
    return image_color


def maxminnorm(arr):
    t = []
    for i in range(len(arr)):
        # print(arr[i])
        # print(max(arr))
        # print(min(arr))
        t.append(float(arr[i] - min(arr))/(max(arr)- min(arr)))
    return t

def textureRegionSelection(pathList):
    
    imgList = []
    heightList = []
    imgAlignTextureList = []
    transformHlist = []
    voteRegion = []
    textureCropList = []
    
    for i in range(len(pathList)):
        imgList.append(cv2.imread(pathList[i]))
        heightList.append(imgList[i].shape[1])
        # 1.运动物体移除
        imgList[i] = motionObjectRemove(imgList[i])
    minIndex = heightList.index(min(heightList))
    # 2.选取最小的图片作为基准
    minImg = imgList[minIndex].copy()
    height, width = minImg.shape[:2]
    bnd = min(height,width)//4
    w = min(height,width)//5
    step = w//20
    # 3.其他每幅图片与最小的图片进行特征点对齐
    for i in range(len(pathList)):
        imAlign, h = alignImages(imgList[i], minImg)
        imgAlignTextureList.append(textureMap(imAlign[bnd:height-bnd,bnd:width-bnd,:]))
        transformHlist.append(h)

    # 4.计算纹理图、深度图、显著图
    ht, wd = imgAlignTextureList[0].shape[:2]
    atextureMap = textureMap(minImg[bnd:height-bnd,bnd:width-bnd,:])
    depthMap = ImgDepthEstimation(minImg[bnd:height-bnd,bnd:width-bnd,:])
    depthMap = cv2.cvtColor(depthMap,cv2.COLOR_BGR2GRAY)
    saliencyMap = SalientObjectDetection(minImg[bnd:height-bnd,bnd:width-bnd,:])
    

    textureList = []
    saliencyList = []
    depthList = []
    disList = []
    # 5.遍历每一区域求特征图的像素和、纹理图的距离
    i = 0
    while (i*step+w)<ht:
        j = 0
        while (j*step+w)<wd:
            dis = 0
            for x in range(len(imgAlignTextureList)):
                for y in range(len(imgAlignTextureList)-x-1):
                    #print([x,y+x+1])
                    # a = gis.generate_signature(imgAlignTextureList[x][i*step:i*step+w,j*step:j*step+w])
                    # b = gis.generate_signature(imgAlignTextureList[y+x+1][i*step:i*step+w,j*step:j*step+w])
                    a = (imgAlignTextureList[x][i*step:i*step+w,j*step:j*step+w]).sum()
                    b = (imgAlignTextureList[y+x+1][i*step:i*step+w,j*step:j*step+w]).sum()
                    dis = dis + abs(a-b)#gis.normalized_distance(a, b)
            textureList.append((atextureMap[i*step:i*step+w,j*step:j*step+w]).sum())
            saliencyList.append((saliencyMap[i*step:i*step+w,j*step:j*step+w]).sum())
            depthList.append((depthMap[i*step:i*step+w,j*step:j*step+w]).sum())
            disList.append(dis)
            j = j + 1
        i = i + 1
    print([i,j])
    # 6.归一化后加权求和得出总票数，取票数最高的区域
    textureList = maxminnorm((textureList))
    saliencyList = maxminnorm((saliencyList))
    depthList = maxminnorm((depthList))
    disList = maxminnorm((disList))

    for i in range(len(textureList)):
        voteRegion.append(1.5*textureList[i]+saliencyList[i]+1-depthList[i]+disList[i])
    
    index = voteRegion.index(max(voteRegion))
    y = index//j
    x = index%j

    ymin = y*step+bnd
    ymax = y*step+w+bnd
    xmin = x*step+bnd
    xmax = x*step+w+bnd  

    for i in range(len(imgList)):
        if i == minIndex:
            textureCropList.append(imgList[minIndex][ymin:ymax,xmin:xmax,:])
        else:
            textureCropList.append(textureRegionMatch(minImg,imgList[i],ymin,ymax,xmin,xmax))

    cv2.rectangle(minImg, (xmin,ymin), (xmax,ymax),(0, 255, 0),thickness = 10, lineType = 4) 

    return minImg,atextureMap,depthMap,saliencyMap,textureCropList


# rootPath = '../dataset_p2/NEW/'
rootPath = 'D:/ErisLU/MyDiplomaProject/Data/pic/'

for picPath in os.listdir(rootPath):
    pathList = []
    start = time.time()
    for pic in os.listdir(rootPath+picPath):
        pathList.append(rootPath+picPath+'/'+pic)
    showImg,atextureMap,depthMap,saliencyMap,textureCropList = textureRegionSelection(pathList)
    end = time.time()
    print(picPath)
    # outShowPath = '../../dataset/Texture/showImg/'
    # outShowPath = '../Texture20/'
    # if not os.path.exists(outShowPath):
    #         os.makedirs(outShowPath)
    # # cv2.imwrite(outShowPath + picPath + '.png',showImg)
    # cv2.imwrite(outShowPath + picPath + '_T.png',atextureMap)
    # cv2.imwrite(outShowPath + picPath + '_D.png',depthMap)
    # cv2.imwrite(outShowPath + picPath + '_S.png',saliencyMap)
    print("Duration: %.2f seconds." % (end - start))
    for i in range(len(textureCropList)):
        outpath = 'Test1/' + picPath
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        cv2.imwrite(outpath+'/'+pathList[i].split('/')[-1].replace('jpg','png'),textureCropList[i])