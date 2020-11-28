### Libraries ####
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from scipy.spatial import Delaunay
from skimage.draw import polygon
from scipy.interpolate import RectBivariateSpline
import skimage.transform as sktr


"""
This section of code is for aligning images!
START
"""
def get_points_align(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, multichannel=True)
    else:
        im2 = sktr.rescale(im2, 1./dscale, multichannel=True)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta


def align_images(im1, im2):
    pts = get_points_align(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

"""
END.
"""







#### Part 1: Correspondences ####
"""
get_points returns 75 + 4 points labeled on im1 and im2, respectively. Used for morph faces.
"""
def get_points(im1, im2):
    assert im1.shape == im2.shape

    corners = [(0, 0), (im1.shape[1], 0), (0, im1.shape[0]), (im1.shape[1], im1.shape[0])]

    print('Please select 75 points in each image for matching.')
    plt.imshow(im1)
    im1_pts = plt.ginput(75, timeout=0)

    plt.close()
    plt.imshow(im2)
    im2_pts = plt.ginput(75, timeout=0)
    plt.close()
    return np.array(corners + im1_pts), np.array(corners + im2_pts)


"""
match_img_size makes sure that both images are the same size before we morph.
"""
def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    print(im1.shape)
    print(im2.shape)
    assert im1.shape == im2.shape
    return im1, im2


#### Part 2: Computing the midway photo ####

"""
Transform_face is responsible for transforming the physical shape of a face from the points labeled
in face_tri_pts to midway_pts. The triangles are determined by tri, which is passed in.
"""
def transform_face(face, face_tri_pts, midway_pts, tri):

    face = face.copy()
    r, g, b = face[:,:,0], face[:,:,1], face[:,:,2]

    f_r = RectBivariateSpline(range(len(face)), range(len(face[0])), r)
    f_g = RectBivariateSpline(range(len(face)), range(len(face[0])), g)
    f_b = RectBivariateSpline(range(len(face)), range(len(face[0])), b)

    def inv_warp(rr, cc, A_inv, func):
        M = np.stack((cc, rr, np.ones((len(rr)))))
        dest = np.matmul(A_inv, M)
        cc, rr = dest[0] + dest[2], dest[1] + dest[2]
        return func(rr, cc, grid=False)

    def computeAffine(tril1_pts, tril2_pts):
        M_1 = np.vstack((np.transpose(np.stack(tril1_pts)), np.ones((1, 3))))
        M_2 = np.vstack((np.transpose(np.stack(tril2_pts)), np.ones((1, 3))))
        return np.matmul(M_2, np.linalg.inv(M_1))

    for triangle in tri.simplices:
        i, j, k = triangle
        face_tri = np.array([face_tri_pts[i], face_tri_pts[j], face_tri_pts[k]])
        midway_tri = np.array([midway_pts[i], midway_pts[j], midway_pts[k]])

        # fails if no mapping is needed, ie matrix A is singular. in which case we don't need to do anything.
        try:
            A = computeAffine(face_tri, midway_tri)
            A_inv = np.linalg.inv(A)

            rr, cc = polygon(midway_tri[:,1], midway_tri[:,0])

            r[rr, cc] = inv_warp(rr, cc, A_inv, f_r)
            g[rr, cc] = inv_warp(rr, cc, A_inv, f_g)
            b[rr, cc] = inv_warp(rr, cc, A_inv, f_b)
        except np.linalg.LinAlgError:
            pass

    # normalize all channels & stack them on each other.
    face = np.dstack((r/r.max(), g/g.max(), b/b.max()))

    ## UNCOMMENT CODE FOR TRIANGULATION VISUALIZATION. ##
    # plt.triplot(midway_pts[:,0], midway_pts[:,1], tri.simplices)
    # plt.plot(midway_pts[:,0], midway_pts[:,1], 'o')
    # plt.imshow(face)
    # plt.show()
    return face


#### Part 3: The morph sequence ####
"""
The morph function first transforms the shape of both images to one determined by the warped_shape.
We then compute a weighted average of both photos.
"""
def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    warped_shape = (1-warp_frac) * im1_pts + warp_frac * im2_pts
    im1 = transform_face(im1, im1_pts, warped_shape, tri)
    im2 = transform_face(im2, im2_pts, warped_shape, tri)
    return (1-dissolve_frac) * im1 + dissolve_frac * im2

"""
Higher order function defined for the entire morphing process. Used to generate the frames for the final gif.
"""
def morph_between_two_photos(p1, p2):
    face_1 = skio.imread(p1)
    face_2 = skio.imread(p2)

    face_1, face_2 = match_img_size(face_1, face_2)
    F1_POINTS, F2_POINTS = get_points(face_1, face_2)

    print(F1_POINTS)
    print(F2_POINTS)
    # F1_POINTS = np.array([(0, 0), (447, 0), (0, 611), (447, 611), (184.84144213994966, 262.9719906958713), (278.2607094398139, 262.9719906958713), (157.65594107385152, 265.44339988369836), (178.91006008916457, 251.6035084318666), (200.16417910447763, 259.0177359953479), (259.47799961232806, 262.47770885830585), (278.7549912773794, 251.6035084318666), (300.0091102926924, 260.50058150804415), (300.5033921302579, 271.86906377204883), (283.2035278154682, 275.32903663500673), (262.4436906377206, 271.86906377204883), (199.66989726691224, 274.83475479744135), (178.91006008916457, 275.82331847257217), (158.6445047489824, 273.8461911223105), (214.9926342314402, 282.743264198488), (212.52122504361319, 299.0545648381469), (208.5669703430898, 331.6771661174646), (245.63810816049624, 282.24898236092264), (253.05233572397754, 300.5374103508432), (257.5008722620663, 330.6886024423338), (224.38398914518325, 322.78009304128705), (235.75247140918782, 322.2858112037216), (196.70420624151973, 369.7368676100019), (227.84396200814115, 359.3569490211281), (265.4093816631131, 369.2425857724365), (215.4869160690057, 390.49670478774954), (241.68385345997285, 390.49670478774954), (234.76390773405706, 57.35074626865662), (293.0891645667766, 68.7192285326613), (351.41442139949606, 116.66456677650706), (343.5059119984494, 256.5463268075208), (350.4258577243653, 297.0774374878852), (328.1831750339213, 333.16001163016085), (314.837565419655, 370.72543128513274), (276.77786392711766, 416.19936034115136), (228.83252568327202, 434.98207016863734), (178.4157782515992, 414.7165148284551), (144.80461329715064, 378.63394068617947), (132.94184919558057, 337.6085481682496), (114.1591393680946, 299.5488466757123), (112.67629385539834, 256.0520449699554), (106.74491180461331, 124.57307617755373), (150.24171351037023, 72.1792013956192), (147.77030432254315, 187.34686954836206), (185.33572397751504, 160.16136848226392), (274.30645473929064, 160.65565031982942), (315.3318472572205, 194.76109711184336), (162.59875944950574, 326.240065904245), (302.97480131808493, 321.2972475285908), (229.3268075208374, 207.61242488854424)])
    # F2_POINTS = np.array([(0, 0), (447, 0), (0, 611), (447, 611), (149.1790075596046, 250.61494475673578), (299.93496801705754, 248.63781740647408), (115.56784260515605, 243.69499903081987), (150.16757123473542, 232.82079860438068), (170.4331265749176, 260.00629967047877), (282.6351037022678, 250.61494475673578), (305.3720682302772, 227.38369839116103), (335.52326032176774, 238.75218065516572), (331.56900562124434, 254.07491761969368), (309.82060476836597, 274.3404729598759), (289.06076759061824, 274.3404729598759), (163.5131808490018, 279.2832913355301), (136.8219616204691, 279.2832913355301), (120.01637914324482, 262.47770885830585), (191.1929637526652, 314.8715836402403), (178.83591781352976, 371.2197131226982), (181.80160883892228, 445.8562705950766), (266.323803062609, 309.4344834270207), (271.76090327582864, 363.8054855592169), (277.69228532661367, 441.9020158945532), (206.51570071719328, 440.41917038185693), (251.4953479356464, 436.4649156813336), (159.06464431091297, 479.9617173870905), (227.76981973250628, 462.6618530723008), (301.9120953673192, 470.0760806357821), (204.04429152936615, 492.31876332622596), (252.97819344834267, 491.3301996510952), (215.9070556309362, 50.430800542740826), (259.8981391742586, 54.8793370808296), (293.01502229114163, 66.74210118239967), (320.2005233572398, 82.06483814692763), (405.21699941849187, 33.62521806551649), (407.1941267687535, 170.54128707113773), (430.42537313432825, 337.6085481682496), (357.7659430122116, 471.55892614847835), (232.21835627059505, 529.3899011436324), (100.24510564062803, 466.12182593525876), (16.711475092072135, 323.76865671641787), (40.437003295212264, 162.13849583252568), (36.48274859468893, 34.61378174064737), (124.4649156813336, 80.58199263423137), (143.74190734638495, 67.23638301996505), (178.83591781352976, 55.86790075596036), (128.91345221942237, 168.5641597208761), (173.3988176003101, 133.47014925373128), (253.96675712347354, 126.55020352781537), (300.9235316921884, 158.67852296956767), (88.38234153905796, 326.240065904245), (379.0200620275246, 309.9287652645861), (222.33271951928663, 185.3697421981003)])
    tri = Delaunay((F1_POINTS + F2_POINTS) / 2)

    # N frames.
    N = 40
    for i in range(N):
        j = i / N
        skio.imsave("test3/frame_"+str(i)+".jpg", morph(face_1, face_2, F1_POINTS, F2_POINTS, tri, j, j))


#### Part 4: The mean face of a population ####

"""
I computed mean faces of two different datasets. As such, I've written two different sets of functions
to parse data as they are formatted differently. All in all, they achieve the same thing.
"""

### Danish Dataset of Computer Scientists ###
F_SUFFIX = '1m'

### Read the labelled points corresponding to Danish Computer Scientists. ###
def read_asf(file):
    i, skip = 0, 16
    res = [[0, 0], [640, 0], [0, 480], [640, 480]]
    for line in file.readlines():
        if i < skip:
            i += 1
            continue
        if i < 16 + 58:
            line = line.split()
            res.append([float(line[2])*640, float(line[3])*480])
        i += 1
    return np.array(res)

### Calculate the mean shape of danish scientists. Also creates a mapping (scientist)->(scientist's labeled points) ###
### Mainly used as helper function to mean_danes() ###
def mean_face():
    n = 0
    face = np.zeros((62, 2))

    cache = {}

    for file in os.scandir('imm_face_db'):
        if file.name.endswith(F_SUFFIX + '.asf'):
            res = read_asf(open(file.path))
            cache[file.name[:len(file.name)-4]] = res
            face = face + res
            n += 1

    face = face/n

    return cache, face

### Calculates the mean face of danish computer scientists. ###
def mean_danes():
    face_pts, mean = mean_face()
    tri = Delaunay(mean)
    count = 0
    res = None
    for file in os.scandir('imm_face_db'):
        if file.name.endswith(F_SUFFIX + '.jpg'):
            im = skio.imread(file.path)
            plt.imshow(im)

            im = transform_face(im, face_pts[file.name[:len(file.name)-4]], mean, tri)
            if res is None:
                res = im
            else:
                res = res + im
            count += 1

    res = res / count
    skio.imshow(res)
    skio.show()
    return mean, res

### Read the labelled points off another dataset I found. ###
def read_tem(file):
    res = [[0, 0], [0, 1349], [1349, 0], [1349, 1349]]
    lines = file.readlines()
    for i in range(1, 190):
        line = lines[i].split()
        res.append([float(line[0]), float(line[1])])
    return np.array(res)

### Find the shape of a group passed in as parameter, along with a mapping. Analagous to mean_face() but for this dataset. ###
def mean_groups_face(group):
    key = csv.reader(open('neutral_front/london_faces_info.csv'))
    cache = {}
    face = np.zeros((193, 2))
    n = 0
    for line in key:
        if line[3] == group:
            face_pts = read_tem(open('neutral_front/'+line[0][1:]+'_03.tem'))
            cache[line[0][1:]+'_03'] = face_pts
            face = face + face_pts
            n += 1

    face = face / n
    return cache, face

### Analagous to mean_danes() ###
def mean_groups(face_pts, mean):
    tri = Delaunay(mean)
    count = 0
    res = np.zeros((1350,1350,3))
    for file in os.scandir('neutral_front'):
        if file.name.endswith('_03.jpg') and file.name[:len(file.name)-4] in face_pts:
            im = skio.imread(file.path)
            im = transform_face(im, face_pts[file.name[:len(file.name)-4]], mean, tri)
            res = res + im
            count += 1

    res = res / count
    return res

### Takes in an image of Brian and the labeled points of Brian and computes a caricature. ###
def caricature(brian, brian_pts):

    _, mmean = mean_groups_face('east_asian')
    mean_male = skio.imread('results/average_male.png')/255.

    mean_male, brian2 = align_images(mean_male, brian.copy())

    dappear, dshape = brian2 - mean_male, brian_pts - mmean
    tri = Delaunay(brian_pts + 0.5*dshape)
    brian = transform_face(brian, brian_pts, brian_pts+0.5*dshape, tri)
    brian, dappear = align_images(brian, dappear)
    test = brian + 2*dappear
    skio.imshow(test)
    skio.show()


def make_female(brian, brian_pts):
    _, mmean = mean_groups_face('male')
    _, fmean = mean_groups_face('female')

    #FOR COMPUTING AVERAGE FEMALE FACE
    #skio.imsave('./average_female.png', mean_groups(face_pts, mean))
    mean_male = skio.imread('results/average_male.png')/255.
    mean_female = skio.imread('results/average_female.png')/255.

    mean_female, mean_male = align_images(mean_female, mean_male)

    # Compute vectors for appearance and shape in direction of female.
    dappear, dshape = mean_male - mean_female, mmean - fmean

    tri = Delaunay(brian_pts + 1.2*dshape)
    brian = transform_face(brian, brian_pts, brian_pts+1.2*dshape, tri)

    brian, dappear = align_images(brian, dappear)
    test = brian + 1.2*dappear
    skio.imshow(test)
    skio.show()

def pca_caricature(brian, brian_pts):
    pass

if __name__ == '__main__':

    #morph_between_two_photos('face_1.jpg', 'tiger.jpg')

    brian = skio.imread('face_1.jpg')/255.
    brian_pts = np.array([[0, 0], [0, 1349], [1349, 0], [1349, 1349], (587.8415070880394, 573.6574279201691), (795.0304885418092, 574.4081126355813), (585.5894529418027, 561.6464724735738), (569.8250739181464, 568.4026349122836), (569.8250739181464, 581.1642750742911), (578.0826057876807, 590.9231763746499), (588.5921918034516, 592.4245458054743), (597.6004083883981, 587.920437513001), (603.6058861116958, 581.9149597897034), (603.6058861116958, 569.904004343108), (793.5291191109848, 563.8985266198104), (776.263370656504, 570.6546890585203), (777.0140553719161, 582.6656445051156), (786.7729566722749, 593.9259152362987), (797.2825426880457, 593.9259152362987), (807.0414439884045, 588.6711222284133), (809.2934981346411, 583.4163292205278), (809.2934981346411, 570.6546890585203), (539.797685301658, 587.1697527975889), (553.3100101790777, 571.4053737739325), (587.8415070880394, 561.6464724735738), (614.8661568428789, 572.1560584893447), (627.6277970048865, 589.4218069438255), (756.7455680557865, 593.1752305208865), (774.0113165102673, 568.4026349122836), (795.0304885418092, 563.8985266198104), (824.3071924428854, 567.6519501968714), (843.8249950436027, 584.1670139359401), (559.3154879023755, 597.6793388133598), (581.0853446493295, 601.4327623904209), (611.1127332658178, 598.430023528772), (770.2578929332062, 601.4327623904209), (807.0414439884045, 602.183447105833), (830.312670166183, 596.9286540979476), (520.2798827009406, 578.9122209280546), (546.5538477403678, 565.3998960506348), (586.340137657215, 550.3862017423907), (621.6223192815888, 548.134147596154), (653.1510773289016, 563.8985266198104), (748.4880361862522, 565.3998960506348), (777.7647400873284, 554.1396253194516), (806.2907592729923, 554.1396253194516), (835.5674631740685, 561.6464724735738), (855.8359504901981, 581.1642750742911), (560.8168573331999, 641.2190523072679), (615.6168415582911, 625.4546732836116), (656.9045009059627, 600.6820776750086), (728.9702335855347, 591.6738610900621), (752.9921444787254, 623.2026191373749), (785.2715872414504, 647.2245300305656), (662.9099786292603, 542.1286698728563), (674.9209340758557, 570.6546890585203), (673.4195646450313, 599.1807082441842), (665.9127174909091, 637.4656287302068), (659.9072397676115, 676.5012339316418), (689.9346283840998, 723.0436862871987), (727.4688641547103, 538.3752462957953), (710.9538004156417, 571.4053737739325), (712.4551698464661, 602.183447105833), (720.7127017160004, 634.462889868558), (719.9620170005883, 675.7505492162295), (630.6305358665353, 683.2573963703517), (624.6250581432377, 708.0299919789545), (646.3949148901917, 735.054641733794), (665.9127174909091, 732.0519028721452), (655.4031314751383, 707.2793072635424), (742.4825584629546, 675.7505492162295), (757.4962527711987, 698.2710906785958), (737.9784501704812, 730.5505334413208), (717.7099628543516, 726.7971098642597), (729.720918300947, 703.5258836864813), (506.76755782352075, 537.6245615803831), (528.5374145704749, 504.5944341022458), (562.3182267640243, 494.08484808647495), (591.5949306651004, 495.58621751729936), (648.6469690364283, 516.6053895488412), (631.3812205819476, 545.1314087345052), (742.4825584629546, 529.3670297108488), (746.2359820400155, 500.84101052518486), (787.5236413876871, 492.58347865565054), (832.5647243124196, 493.3341633710627), (866.345536505969, 524.1122367029633), (876.1044378063277, 540.6273004420319), (548.8059018866045, 527.1149755646121), (594.5976695267493, 527.1149755646121), (787.5236413876871, 529.3670297108488), (836.3181478894807, 528.6163449954365), (611.8634179812301, 816.8792757137248), (639.6387524514819, 801.1148966900685), (666.6634022063214, 790.6053106742976), (690.6853130995121, 798.1121578284196), (713.9565392772905, 788.3532565280609), (744.7346126091911, 807.1203744133661), (770.2578929332062, 815.3779062829004), (641.1401218823063, 819.8820145753737), (668.915456352558, 819.8820145753737), (690.6853130995121, 823.6354381524347), (710.2031157002295, 822.1340687216103), (737.9784501704812, 822.1340687216103), (641.8908065977184, 830.3916005911447), (666.6634022063214, 829.6409158757324), (686.931889522451, 829.6409158757324), (710.9538004156417, 829.6409158757324), (740.2305043167179, 829.6409158757324), (622.373003997001, 840.9011866069155), (639.6387524514819, 855.1641961997475), (687.6825742378633, 865.6737822155185), (736.4770807396568, 851.4107726226864), (755.9948833403743, 835.64639359903), (452.7182583138417, 548.134147596154), (459.4744207525516, 655.4820619000999), (471.4853761991469, 735.054641733794), (939.1619539009533, 551.8875711732151), (919.6441513002359, 641.9697370226801), (898.624979268694, 729.7998487259086), (427.94566270523876, 566.9012654814592), (424.1922391281778, 614.1944025524284), (436.95387929018534, 655.4820619000999), (443.71004172889525, 697.5204059631836), (460.2251054679638, 735.054641733794), (950.4222246321365, 570.6546890585203), (965.4359189403807, 607.4382401137185), (951.9235940629609, 656.2327466155122), (940.6633233317777, 684.0080810857638), (925.6496290235335, 733.5532723029696), (497.008656523162, 812.3751674212516), (514.2744049776429, 864.9230975001062), (567.5730197719097, 915.9696581481364), (614.8661568428789, 942.2436231875638), (684.6798353762144, 964.7641646499301), (766.5044693561453, 936.9888301796783), (819.0523994349999, 899.4545944090678), (860.3400587826713, 843.1532407531522), (894.1208709762208, 783.8491482355877), (475.238799776208, 497.08758694812377), (501.51276481563536, 470.8136219086964), (501.51276481563536, 407.00542109865864), (571.3264433489708, 350.70406744274294), (634.3839594435964, 353.70680630439176), (693.6880519611609, 352.20543687356735), (773.260631794855, 347.7013285810941), (844.575679759015, 376.22734776675804), (879.1071766679765, 409.2574752448953), (873.8523836600912, 470.8136219086964), (909.8852499998771, 506.09580353307024), (518.7785132701162, 937.7395148950906), (402.42238238122366, 775.5916163660534), (381.4032103496818, 524.1122367029633), (396.416904657926, 372.4739241896971), (448.96483473678063, 229.09314354596518), (552.5593254636656, 162.28220387427837), (699.6935296844586, 129.25207639614132), (846.8277339052516, 169.03836631298827), (954.1756482091976, 253.86573915456802), (981.2002979640371, 371.7232394742848), (999.9674158493423, 533.871138003322), (957.1783870708464, 784.5998329509998), (864.8441670751446, 933.9860913180295), (616.3675262737033, 712.5341002714277), (584.0880835109783, 755.3231290499238), (564.5702809102609, 801.8655814054807), (767.2551540715574, 711.0327308406033), (782.2688483798016, 743.3121736033283), (793.5291191109848, 793.6080495359464), (484.9977010765667, 652.4793230384511), (536.0442617245969, 672.7478103545807), (590.093561234276, 685.5094505165882), (905.3811417074039, 641.9697370226801), (855.0852657747859, 659.9861701925731), (792.7784343955725, 673.4984950699929), (670.4168257833825, 744.0628583187406), (666.6634022063214, 766.5833997811069), (710.2031157002295, 747.0655971803894), (710.2031157002295, 768.0847692119313), (689.9346283840998, 897.2025402628312), (687.6825742378633, 930.9833524563807), (655.4031314751383, 898.7039096936556), (686.6723935952809, 885.8520546668963), (722.2140711468248, 897.9532249782435), (676.4223035066801, 706.5286225481301), (683.9291506608022, 716.2875238484888), (704.1976379769318, 718.5395779947255), (713.9565392772905, 703.5258836864813), (629.8798511511231, 1014.3093558671359), (746.9866667554278, 1014.3093558671359), (457.22236660631495, 616.446456698665), (468.4826373374981, 701.2738295402446), (922.6468901618847, 618.6985108449016), (907.6331958536406, 702.0245142556569)])
    #caricature(brian, brian_pts)
    #make_female(brian, brian_pts)
    mean_danes()


