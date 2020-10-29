import cv2 as cv

def get_ROI(_img):
    img = _img
    for i in range(4):
        roi = cv.selectROI('roi', img, True, False)
        x, y, w, h = roi
        img[y:y+h, x:x+w] = [0,0,0]
    cv.destroyAllWindows()
    return img

img = cv.imread('dataset/-1-0-0/images/-1-0-0.jpg')
img = get_ROI(img)
cv.imwrite('./-1-0-0-test.jpg', img)