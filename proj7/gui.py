import numpy as np
import cv2

def determine_vertex(x, y, rect, dp):
    u = 10
    if x in range(rect[0][0]-u, rect[0][0]+u) \
        and y in range(rect[0][1]-u, rect[0][1]+u):
        return 0

    elif x in range(rect[1][0]-u, rect[1][0]+u) \
        and y in range(rect[1][1]-u, rect[1][1]+u):
        return 1

    elif x in range(dp[0]-u, dp[0]+u) \
        and y in range(dp[1]-u, dp[1]+u):
        return 2

    return -1


def render(img):
    img = img.copy()
    img = cv2.rectangle(img, RECT_COORDS[0], RECT_COORDS[1],(0,255,0),3)
    img = cv2.circle(img, DISSAPEARING_PT, 7, (0, 0, 255), 3)

    corners = [
        RECT_COORDS[0],
        (RECT_COORDS[0][0], RECT_COORDS[1][1]),
        RECT_COORDS[1],
        (RECT_COORDS[1][0], RECT_COORDS[0][1])
        ]

    for i in range(len(corners)):
        corner = corners[i]

        m = (corner[1] - DISSAPEARING_PT[1]) / (corner[0] - DISSAPEARING_PT[0])
        b = DISSAPEARING_PT[1] - m * DISSAPEARING_PT[0] 
        if i < 2:
            # draw to x = 0
            x = 0
        else:
            # draw to x = imwidth
            x = img.shape[1]

        img = cv2.line(img, DISSAPEARING_PT, (x, int(m*x + b)), (0, 255, 0), 3)

    return img



def mouse_responder(event, x, y, flags, param):
    global IS_MOUSE_DOWN, RECT_COORDS, VERTEX, DISSAPEARING_PT


    if event == cv2.EVENT_MOUSEMOVE and IS_MOUSE_DOWN:
        if VERTEX in {0, 1}:
            RECT_COORDS[VERTEX] = (x, y)
        elif VERTEX == 2:
            DISSAPEARING_PT = (x, y)

        layer = render(img)
        cv2.imshow('image', layer)

    elif event == cv2.EVENT_LBUTTONDOWN:
        IS_MOUSE_DOWN = True
        VERTEX = determine_vertex(x, y, RECT_COORDS, DISSAPEARING_PT)
    elif event == cv2.EVENT_LBUTTONUP:
        IS_MOUSE_DOWN = False
        VERTEX = -1


def run_gui():

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_responder)
    cv2.imshow('image', render(img))

    while True:

        key = chr(cv2.waitKey(0))
        if key == 'q':
            print("\nPress q again if you are sure you want to quit. Anything else otherwise.\n")
            confirm = chr(cv2.waitKey(0))
            if confirm == 'q':
                print("quitting. \n")
                break
            else:
                print("not quitting. \n")

        print(RECT_COORDS)


        cv2.imshow('image', img)

        

if __name__ == '__main__':
    img = cv2.imread('dutch.jpg')

    IS_MOUSE_DOWN = False
    RECT_COORDS = [(img.shape[0] // 4, img.shape[1] // 4), (3 * img.shape[0] // 4, 3 * img.shape[1] // 4)]
    VERTEX = -1
    DISSAPEARING_PT = (img.shape[0] // 2, img.shape[1] // 2)

    run_gui()
    cv2.destroyAllWindows()

