import skvideo.io
import skimage.io
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_3D_points():
    points = []
    for z in range(3):
        if z < 2:
            for y in range(3):
                points.append((0, y, z))
            for x in range(1, 5):
                points.append((x, 0, z))
        else:
            for y in range(3):
                for x in range(5):
                    points.append((x, y, z))
    return np.multiply(np.array(points), 100)


def label_points(videodata):
    plt.imshow(videodata[0])
    points = plt.ginput(n=29, timeout=0)
    bboxes = [(point[0], point[1], 10, 10) for point in points]
    print(bboxes)
    return bboxes


def generate_label_video(bboxes, videodata):
    trackers = []
    for bbox in bboxes:
        tracker = cv2.TrackerMedianFlow_create()
        ok = tracker.init(videodata[0], bbox)
        if ok:
            trackers.append(tracker)

    for frame in videodata:
        for i in range(len(trackers)):
            tracker = trackers[i]
            ok, bbox = tracker.update(frame)

            if ok:
                coords = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                cv2.circle(frame, coords, 10, (0, 255, 0), 5)

    skvideo.io.vwrite("./outputvideo.mp4", videodata)


def compute_P(TD_pts, THD_pts):
    assert len(TD_pts) == len(THD_pts)
    N = len(TD_pts)
    A = []
    for i in range(N):
        X, Y, Z = THD_pts[i]
        x, y = TD_pts[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    A = np.array(A)
    w, v = np.linalg.eig(np.matmul(A.T, A))
    p = v[:,11]
    return p.reshape((3, 4))


def draw(img, corners, imgpts):
    """
    Adapted from tutorial: 
    https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html
    """
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img
    

def project(P, cube):
    cube = np.vstack((cube.T, np.ones(cube.shape[0])))
    pts = np.matmul(P, cube)
    pts[0] /= pts[2]
    pts[1] /= pts[2]
    pts[2] /= pts[2]
    rv = pts[:2].T
    return rv


def AR(videodata, bboxes, THD_pts, cube):
    trackers = []
    for bbox in bboxes:
        tracker = cv2.TrackerMedianFlow_create()
        ok = tracker.init(videodata[0], bbox)
        if ok:
            trackers.append(tracker)

    for f in range(len(videodata)):
        frame = videodata[f]
        TD_pts = []
        for i in range(len(trackers)):
            tracker = trackers[i]
            ok, bbox = tracker.update(frame)
            coords = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            cv2.circle(frame, coords, 10, (0, 255, 0), 5)
            TD_pts.append(coords)

        P = compute_P(TD_pts, THD_pts)
        imgpts = project(P, cube)
        frame = draw(frame, None, imgpts)
        # skimage.io.imshow(frame)
        # skimage.io.show()
        # return

    skvideo.io.vwrite("./testvideo.mp4", videodata)


if __name__ == '__main__':
    videodata = skvideo.io.vread('test4.MOV')
    bboxes = [(514.016257867265, 1198.2340761041714, 10, 10), (409.5692732506093, 1150.6379818484804, 10, 10), (310.4107435512524, 1105.6861150514385, 10, 10), (570.8671482282296, 1118.9072523446862, 10, 10), (626.3959248598694, 1036.9362011265512, 10, 10), (666.0593367396123, 993.3064480588341, 10, 10), (712.3333172659786, 923.2344204046219, 10, 10), (532.5258500778116, 1104.3640013221138, 10, 10), (417.5019556265579, 1072.6332718183196, 10, 10), (310.4107435512524, 1022.3929501039788, 10, 10), (586.7325129801268, 1036.9362011265512, 10, 10), (642.2612896117666, 972.152628389638, 10, 10), (687.2131564088083, 913.9796242993486, 10, 10), (725.5544545592263, 857.1287339383841, 10, 10), (549.7133285590335, 1005.2054716227569, 10, 10), (599.9536502733743, 943.0661263444933, 10, 10), (651.5160857170399, 876.9604398782556, 10, 10), (696.4679525140816, 818.7874357879662, 10, 10), (737.4534781231491, 769.86922780295, 10, 10), (434.68943410777956, 965.5420597430143, 10, 10), (494.18455192739384, 891.5036909008279, 10, 10), (549.7133285590335, 829.3643456225641, 10, 10), (595.9873090854, 772.5134552615996, 10, 10), (642.2612896117666, 724.9173610059083, 10, 10), (326.2761083031496, 921.9123066752973, 10, 10), (388.4154535814132, 849.1960515624355, 10, 10), (445.2663439423777, 787.0567062841719, 10, 10), (498.15089311536804, 735.4942708405065, 10, 10), (545.7469873710593, 690.5424040434646, 10, 10)]
    THD_pts = get_3D_points()
    cube = np.array([(0, 1, 2), (0, 2, 2), (2, 2, 2), (2, 1, 2), (0, 1, 3), (0, 2, 3), (2, 2, 3), (2, 1, 3)])
    cube = np.multiply(cube, 100)
    AR(videodata, bboxes, THD_pts, cube)


