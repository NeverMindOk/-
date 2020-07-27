from keras.applications import InceptionV3, inception_v3
from keras.layers import Lambda, Dropout, Dense
from keras.models import Input, Model

model_image_size = (240, 240)
batch_size = 32
class_num = 5


## 导入模型
def load_model(weights_path):
    x = Input((*model_image_size, 3))
    x = Lambda(inception_v3.preprocess_input)(x)

    base_model = InceptionV3(input_tensor=x,
                             weights='imagenet',
                             include_top=False,
                             pooling='avg',
                            )

    x = base_model.output
    x = Dropout(rate=0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(class_num, activation='softmax')(x)

    model = Model(base_model.input, x)

    model.load_weights(weights_path)

    print("base_model layer count {}".format(len(base_model.layers)))
    print("total layer count {}".format(len(model.layers)))


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

#
#
# if __name__=='__main__':
#
#     weights_path = 'inceptionV3_tof_1563864540.126342.h5'
#     model = load_model(weights_path)
#     ## 验证集
#     # valid_gen = ImageDataGenerator()
#
#     # valid_dir = ['driver_tof_2/valid',
#     #              'driver_tof_3/test1',
#     #              'driver_tof_3/test2'][1]
#
#     # valid_generator = valid_gen.flow_from_directory(valid_dir,
#     #                                                 model_image_size,
#     #                                                 shuffle=False,
#     #                                                 batch_size=batch_size,
#     #                                                 class_mode="categorical")
#     # steps_valid_sample = len(valid_generator)
#
#     # print("subdior to valid type {}".format(valid_generator.class_indices))
#     # print("valid_generator.samples = {}".format(valid_generator.samples))
#     # print(steps_valid_sample,)
#
#     # y_eval = model.evaluate_generator(valid_generator,
#     #                                   steps=steps_valid_sample,
#     #                                   verbose=1)
#
#
#
#     # print(y_eval)
#     path = r'e:\SONY\Desktop\pyCode\Driver_Action\driver_inception\driver_tof_3\test1\c2/'
#     img_list = os.listdir(path)
#
#     for img_name in img_list:
#         img_dep = cv2.imread(path+img_name, -1).astype('float32')
#         img_dep = cv2.merge([img_dep]*3)
#         img_dep = np.expand_dims(img_dep, axis=0)
#
#
#         y_pred = model.predict(img_dep)
#         predict_label=np.argmax(y_pred,axis=1)
#         print(y_pred, predict_label)