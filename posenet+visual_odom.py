import cv2
import numpy as np
import math
import os
from PIL import Image
from data_loader import get_loader
from solver import Solver

orb = cv2.ORB_create(
                        nfeatures=5000,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        firstLevel=0,
                        WTA_K=2,
                        scoreType=cv2.ORB_FAST_SCORE,
                        patchSize=31,
                        fastThreshold=25,
                        )

bf = cv2.BFMatcher(cv2.NORM_HAMMING)


if __name__ == "__main__":
    focal = 1081
    pp = (356.489, 645.504)
    textOrg1 = (10,30)

    img_path = "./data/test_img/"
    img_list = os.listdir(img_path)
    img_list.sort()
    img_1_c = cv2.imread(img_path + img_list[0])
    img_2_c = cv2.imread(img_path + img_list[1])
    img_1 = cv2.cvtColor(img_1_c,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img_1,None)
    kp2, des2 = orb.detectAndCompute(img_2,None)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    idx = matches[0:1500]

    pts1 = []
    pts2 = []

    for i in idx:
        pts1.append(kp1[i.queryIdx].pt)
        pts2.append(kp2[i.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    print("Estimate pose of " + img_list[0])
    image = Image.open(img_path + img_list[0])
    data_loader = get_loader(image=image, model='Resnet', batch_size=1)
    solver = Solver(data_loader)
    pos, ori = solver.test()
    q0 = ori[0]
    q1 = ori[1]
    q2 = ori[2]
    q3 = ori[3]
    
    t_ff = np.array([[pos[0]],
                    [pos[1]],
                    [pos[2]]])

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    R_ff = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])

    E, mask = cv2.findEssentialMat(pts1,pts2,focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
    _, R_n, t_n, _ = cv2.recoverPose(E, pts1, pts2, focal = focal, pp = pp)

    t_f = t_ff + R_ff.dot(t_n)
    R_f = R_n.dot(R_ff)

    prevImage = img_2
    kp_prev = kp2
    des_prev = des2

    x = t_f[0][0]
    y = t_f[1][0]
    z = t_f[2][0]
    qw = math.sqrt(1+R_f[0][0]+R_f[1][1]+R_f[2][2]) / 2
    qx = (R_f[2][1]-R_f[1][2]) / (4*qw)
    qy = (R_f[0][2]-R_f[2][0]) / (4*qw)
    qz = (R_f[1][0]-R_f[0][1]) / (4*qw)
    posenet_list = np.array([[img_list[0], pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3]]])
    posenet_list = np.append(posenet_list, [[img_list[1],x , y, z, qw, qx, qy, qz]], axis = 0)

    traj = np.zeros((1000,2000),dtype=np.uint8)
    traj = cv2.cvtColor(traj,cv2.COLOR_GRAY2BGR)
    
    for numFrame in range(2, len(img_list)):
        filename = img_path + img_list[numFrame]
        
        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)

        kp_curr, des_curr = orb.detectAndCompute(currImage,None)

        matches = bf.match(des_prev,des_curr)
        matches = sorted(matches, key = lambda x:x.distance)
        idx = matches[0:1500]

        pts1 = []
        pts2 = []

        for i in idx:
            pts1.append(kp_prev[i.queryIdx].pt)
            pts2.append(kp_curr[i.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        E_mat, mask_n = cv2.findEssentialMat(pts2, pts1, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R_cur, t_cur, _ = cv2.recoverPose(E_mat, pts2, pts1, focal = focal, pp = pp)

        t_f = t_f + R_f.dot(t_cur)
        R_f = R_cur.dot(R_f)

        x = t_f[0][0]
        y = t_f[1][0]
        z = t_f[2][0]
        qw = math.sqrt(1+R_f[0][0]+R_f[1][1]+R_f[2][2]) / 2
        qx = (R_f[2][1]-R_f[1][2]) / (4*qw)
        qy = (R_f[0][2]-R_f[2][0]) / (4*qw)
        qz = (R_f[1][0]-R_f[0][1]) / (4*qw)
        posenet_list = np.append(posenet_list, [[img_list[numFrame], x, y, z, qw, qx, qy, qz]], axis = 0)

        prevImage = currImage
        kp_prev = kp_curr
        des_prev = des_curr

        x = int(t_f[0]) + 1000
        y = int(t_f[2]) + 100

        cv2.circle(traj, (x,y), 1 , (0,255,0), 2)
        
        cv2.rectangle(traj, (10,10), (700,150), (0,0,0), -1)
        text1 = 'Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f[0]),float(t_f[1]),float(t_f[2]))
        cv2.putText(traj, text1, textOrg1, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)

        feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)

        cv2.imshow("trajectory", traj)
        cv2.imshow("feat_img", feature_img)

        cv2.waitKey(1)

    cv2.imwrite("./result/pose+odom_result.png",traj)

    with open("./result/pose+odom_result.txt", "w") as f:
        f.write("Pose\n")
        f.write("x y z qw qx qy qz\n")
        f.write("---------------\n")
        for line in posenet_list:
            result = ""
            for j in line:
                result += j + " "
            result += "\n"
            f.write(result)