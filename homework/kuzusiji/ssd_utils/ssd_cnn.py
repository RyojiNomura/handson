from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense
from keras.layers import Flatten, Reshape, Activation, Concatenate, Dropout

from ssd_utils.ssd_box import DefaultBox

class SSD_CNN():
    def __init__(self, num_classes, cnn_size, ssd_size,
                variances=[0.1, 0.1, 0.2, 0.2]):
        self.num_classes = num_classes
        self.cnn_size = cnn_size
        self.ssd_size = ssd_size
        self.img_size = (300, 300, 1) #img_size
        self.variances = variances # variances for box
        self.dim_box = 4 #(cx, cy, w, h)

    def load(self, path):
        '''
        Arg:
            cnn.hdf5 file path
        return:
            ssd model
        '''

        model = self.build_cnn()
        model.load_weights(path)
        return model

    def build_cnn(self):
        """
        build cnn network
        """

        ## Block 1
        self.conv1_1 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv1_1')
        self.conv1_2 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv1_2')
        self.pool1 = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool1')
        
        ## Block 2
        self.conv2_1 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv2_1')
        self.conv2_2 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv2_2')
        self.pool2 = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool2')
        
        ## Block 3
        self.conv3_1 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv3_1')
        self.conv3_2 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv3_2')
        self.pool3 = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool3')
        
        ## Block 4
        self.flat = Flatten(name='flat')
        self.dense1 = Dense(self.num_classes,activation='relu', name='dense1')
        self.dense2 = Dense(self.num_classes, activation='softmax', name='dense2')

        img_size = self.cnn_size
        inputs = Input(shape=img_size, name='cnn_input')

        ## Block 1
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        ## Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        ## Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        
        ## Block cnn_out
        x = self.flat(x)
        x = self.dense1(x)
        outputs = self.dense2(x)

        return Model(inputs, outputs)

    def build_ssd(self):
        ## additional Block 4
        self.conv4 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv4')
        self.pool4 = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool4')

        ## additional Block 5
        self.conv5 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv5')
        self.pool5 = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool5')

        img_size = self.ssd_size
        inputs = Input(shape=img_size, name='ssd_input')

        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        conv3 = self.pool3(x)

        x = self.conv4(conv3)
        conv4 = self.pool4(x)
        
        x = self.conv5(conv4)
        conv5 = self.pool5(x)

        self.detector_layers = [conv3, conv4, conv5]
        pred_SSD = self.detectors()
        return  Model(inputs, pred_SSD)

    def get_detector(self):
        return self.detector_layers
    
    def detectors(self):
        """
        layers: list of layer
        to learn weight for any num_classes, additional '_' is in mbox layer names.
        """
        mbox_loc_list = []
        mbox_conf_list = []
        mbox_defbox_list = []

        num_def = 6 
        aspect_ratios = [2, 3] # -> [1(min), 1((min*max)**0.5), 2, 3, 1/2, 1/3] by DefaultBox()
        
        for layer in self.detector_layers:

            name_layer = layer.name.split('/')[0] + '_' # eg. 'conv5_1/Relu:0'-> 'conv5_1'

            layer_mbox_loc = Conv2D(num_def * self.dim_box,(3,3),padding='same',
                                    name='{}_mbox_loc'.format(name_layer))(layer)
            layer_length = layer_mbox_loc.shape[1].value
            layer_mbox_loc_flat = Flatten(name='{}_mbox_loc_flat'.format(name_layer))(layer_mbox_loc)
            mbox_loc_list.append(layer_mbox_loc_flat)
            
            layer_mbox_conf = Conv2D(num_def * self.num_classes,(3,3),padding='same',
                                    name='{}_mbox_conf'.format(name_layer))(layer)
            layer_mbox_conf_flat = Flatten(name='{}_mbox_conf_flat'.format(name_layer))(layer_mbox_conf)
            mbox_conf_list.append(layer_mbox_conf_flat)
            
            layer_mbox_defbox = DefaultBox(self.img_size,
                                        self.img_size[0]/layer_length*0.8,
                                        self.img_size[0]/layer_length,
                                        aspect_ratios=aspect_ratios,
                                        variances=self.variances,
                                        name='{}_mbox_defbox'.format(name_layer))(layer)
            mbox_defbox_list.append(layer_mbox_defbox)
            
        mbox_loc = Concatenate(name='mbox_loc', axis=1)(mbox_loc_list)
        num_boxes = mbox_loc._keras_shape[-1] // 4
        mbox_loc = Reshape((num_boxes, self.dim_box),name='mbox_loc_final')(mbox_loc)

        mbox_conf = Concatenate(name='mbox_conf', axis=1)(mbox_conf_list)
        mbox_conf = Reshape((num_boxes, self.num_classes),name='mbox_conf_logits')(mbox_conf)
        mbox_conf = Activation('softmax',name='mbox_conf_final')(mbox_conf)
        
        mbox_defbox = Concatenate(name='mbox_defbox',axis=1)(mbox_defbox_list)

        predictions = Concatenate(name='predictions',axis=2)([mbox_loc, mbox_conf, mbox_defbox])
        return predictions
        