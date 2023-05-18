import os
import cv2 as cv
import flask as f
import numpy as np
import tensorflow as tf

app = f.Flask(__name__, template_folder='templates', static_folder='static')
app.config['IMAGE_UPLOADS'] = r'static\media'

def predict_mask(from_path, image_path, image, model_path):
    classes = [[0, 0, 142], [45, 60, 150], [70, 70, 70], [70, 130, 180], [81, 0, 81], [100, 40, 40], [102, 102, 156],
               [107, 142, 35], [110, 190, 160], [128, 64, 128], [145, 170, 100], [150, 100, 100], [153, 153, 153],
               [157, 234, 50], [170, 120, 50], [220, 20, 60], [220, 220, 0], [230, 150, 140], [244, 35, 232],
               [250, 170, 30]]
    if from_path:
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img = image
    img = cv.resize(img, (512, 384))
    img = np.array(img, dtype='float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    model = tf.keras.models.load_model(model_path, compile=False)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    pred = np.argmax(pred, axis=-1)
    pred = np.expand_dims(pred, axis=-1)
    pred = np.array(pred, dtype='int32')
    p = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype='int32')
    for i, color in enumerate(classes):
        index = (pred == i).all(axis=-1)
        p[index] = color
    p = np.array(p, dtype='uint8')
    return p

def predict_bbox(image, mask, min_area, k, t1, t2, colors):
    mask = cv.copyMakeBorder(mask, 3, 3, 3, 3, cv.BORDER_CONSTANT)
    image = cv.copyMakeBorder(image, 3, 3, 3, 3, cv.BORDER_CONSTANT)
    mask = np.array(mask, dtype='int32')
    for color in colors:
        m = np.zeros(shape=mask.shape[:-1], dtype='int32')
        m[(mask == color).all(axis=-1)] = 255
        m = np.stack([m, m, m], axis=-1)
        m = np.array(m, dtype='uint8')
        kernel = np.ones(shape=(k, k))
        m = cv.dilate(m, kernel, iterations=2)
        m = cv.erode(m, kernel, iterations=1)
        canny = cv.Canny(m, t1, t2)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            for i, contour in enumerate(contours):
                area = cv.contourArea(contour)
                if area > min_area:
                    x1, y1, w, h = cv.boundingRect(contour)
                    image = cv.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if f.request.method == 'GET':
        return f.render_template('index.html')
    img = f.request.files['image']
    img_path = os.path.join(app.config['IMAGE_UPLOADS'], img.filename)
    img.save(img_path)
    mask = predict_mask(True, img_path, None, 'weights.h5')
    mask_path = os.path.join(app.config['IMAGE_UPLOADS'], 'mask_' + os.path.basename(img_path))
    cv.imwrite(mask_path, mask)
    mask = mask[:, :, ::-1]
    img = cv.imread(img_path)
    img = cv.resize(img, (512, 384))
    bbox = predict_bbox(img, mask, 35, 3, 10, 100, [[142, 0, 0], [160, 190, 110], [153, 153, 153], [40, 40, 100], [140, 150, 230], [70, 70, 70], [156, 102, 102], [35, 142, 107], [60, 20, 220]])
    bbox_path = os.path.join(app.config['IMAGE_UPLOADS'], 'bbox_' + os.path.basename(img_path))
    cv.imwrite(bbox_path, bbox)
    return f.render_template('index.html', mask_path=mask_path, bbox_path=bbox_path)

if __name__ == '__main__':
    app.run(debug=True)
