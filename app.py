import os
from flask import Flask, render_template, request, send_file
import zipfile
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from werkzeug.utils import secure_filename
from tensorflow import keras
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNetConfig:

    def __init__(self):
        self.encoder_type = "resnet50"
        self.decoder_type = "vanilla"
        self.decoder_block_channels = [256, 128, 64, 32, 16]
        self.segm_class_count = 1
        self.decoder_attention_type = None

        self.original_height = 751
        self.original_width = 280

        self.target_height = 512
        self.target_width = 256


        self.resize_method = "padding"

        self.input_channels = 3

        self.learning_rate = 0.002
        self.weight_decay = 0.000001
        self.batch_size = 24
        self.max_epochs = 30

        self.random_seed = 42
        self.step_size = 10
        self.gamma = 0.1

        logger.info(f"Config initialized:")
        logger.info(f"  Original size: ({self.original_height}, {self.original_width})")
        logger.info(f"  Target size: ({self.target_height}, {self.target_width})")
        logger.info(f"  Method: {self.resize_method}")

class DataPreprocessor:
    def __init__(self, config: UNetConfig):
        self.config = config
        self.original_height = config.original_height
        self.original_width  = config.original_width
        self.target_height   = config.target_height
        self.target_width    = config.target_width

        ratio_h = self.target_height / self.original_height
        ratio_w = self.target_width  / self.original_width
        self.scale = min(ratio_h, ratio_w)

        new_h = int(self.original_height * self.scale)
        new_w = int(self.original_width  * self.scale)

        self.pad_top    = (self.target_height - new_h) // 2
        self.pad_bottom = self.target_height - new_h - self.pad_top
        self.pad_left   = (self.target_width  - new_w) // 2
        self.pad_right  = self.target_width  - new_w - self.pad_left

        self.resized_h = new_h
        self.resized_w = new_w



    def preprocess(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)

        tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        resized = tf.image.resize(tensor, [self.resized_h, self.resized_w],
                                  method='bilinear')
        resized = tf.squeeze(resized, 0)

        padded = tf.pad(
            resized,
            [[self.pad_top, self.pad_bottom],
             [self.pad_left, self.pad_right],
            [0, 0]],
            mode='constant',
            constant_values=0
        )
        return padded.numpy()

    def postprocess_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        
        cropped = pred_mask[
            self.pad_top : self.pad_top + self.resized_h,
            self.pad_left: self.pad_left + self.resized_w
        ]

        tensor = tf.convert_to_tensor(cropped, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=-1) 
        tensor = tf.expand_dims(tensor, 0)       
        original = tf.image.resize(tensor, [self.original_height, self.original_width],
                                   method='bilinear')
        return tf.squeeze(original, [0, -1]).numpy()
    

class Basic2DDecoderBlock(keras.layers.Layer):

    def __init__(self, out_channels, kernel_size=3, **kwargs):
        super(Basic2DDecoderBlock, self).__init__(**kwargs)
        self.out_channels = out_channels

        self.upsampler = keras.layers.Conv2DTranspose(
            out_channels, (2, 2), strides=(2, 2),
            padding='same', kernel_initializer='he_normal'
        )

        self.conv1 = keras.layers.Conv2D(
            out_channels, kernel_size, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.ReLU()

        self.conv2 = keras.layers.Conv2D(
            out_channels, kernel_size, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.ReLU()

    def call(self, prev, skip=None, training=None):
        x = self.upsampler(prev)

        if skip is not None:
            x = keras.layers.concatenate([skip, x], axis=-1)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        return x
    
def build_resnet_encoder(input_shape, weights='imagenet'):
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )

    layer_names = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out',
    ]

    outputs = [base_model.get_layer(name).output for name in layer_names]
    encoder = keras.Model(inputs=base_model.input, outputs=outputs, name='resnet_encoder')

    return encoder

class VanillaDecoder(keras.layers.Layer):

    def __init__(self, decoder_channels=[256, 128, 64, 32, 16],
                 num_classes=1, **kwargs):
        super(VanillaDecoder, self).__init__(**kwargs)

        self.decoder_blocks = [
            Basic2DDecoderBlock(channels)
            for channels in decoder_channels
        ]

        self.head = keras.layers.Conv2D(
            num_classes, (1, 1), activation='sigmoid',
            kernel_initializer='he_normal', name='output_head'
        )

    def call(self, encoder_features, training=None):
        x = encoder_features[-1]
        skip_connections = encoder_features[:-1][::-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, skip=skip, training=training)

        x = self.head(x)
        return x
    
class UNetModel(keras.Model):
    def __init__(self, config: UNetConfig, **kwargs):
        super(UNetModel, self).__init__(**kwargs)
        self.config = config

        input_shape = (config.target_height, config.target_width, config.input_channels)
        self.encoder = build_resnet_encoder(input_shape, weights='imagenet')
        self.decoder = VanillaDecoder(
            decoder_channels=config.decoder_block_channels,
            num_classes=config.segm_class_count
        )

    def call(self, x, training=None):
        encoder_features = self.encoder(x, training=training)
        output = self.decoder(encoder_features, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_height": self.config.target_height,
            "target_width":  self.config.target_width,
            "input_channels": self.config.input_channels,
            "decoder_block_channels": self.config.decoder_block_channels,
            "segm_class_count": self.config.segm_class_count,
        })
        return config

    @classmethod
    def from_config(cls, config):
        height = config.pop("target_height")
        width  = config.pop("target_width")
        channels = config.pop("input_channels")
        dec_channels = config.pop("decoder_block_channels")
        classes = config.pop("segm_class_count")

        dummy_config = UNetConfig()
        dummy_config.target_height = height
        dummy_config.target_width  = width
        dummy_config.input_channels = channels
        dummy_config.decoder_block_channels = dec_channels
        dummy_config.segm_class_count = classes

        return cls(dummy_config, **config)
    


    


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'models/final_model.h5'
config = UNetConfig()
preprocessor = DataPreprocessor(config)

custom_objects = {
    'UNetModel': UNetModel,
    'Basic2DDecoderBlock': Basic2DDecoderBlock,
    'VanillaDecoder': VanillaDecoder,
}

try:
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model directly: {e}")
    print("Attempting to instantiate model and then load weights (less robust for full model save).")
    model = UNetModel(config) 
    dummy_input = tf.zeros((1, config.target_height, config.target_width, config.input_channels))
    _ = model(dummy_input)

    try:
        model.load_weights(model_path)
        print("Model weights loaded successfully using model.load_weights().")
    except Exception as e_weights:
        print(f"Error loading weights with model.load_weights(): {e_weights}")
        print("Manual weight extraction and setting is generally not recommended due to potential ordering issues.")
        print("Please ensure your model architecture in Python matches the saved model.")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'npzfile' not in request.files:
        return "Tidak ada file .npz yang diupload", 400

    file = request.files['npzfile']
    if file.filename == '':
        return "Tidak ada file yang dipilih", 400

    if not file.filename.endswith('.npz'):
        return "File harus .npz", 400

    if file:
        filename = secure_filename(file.filename)
        npz_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(npz_path)

        data = np.load(npz_path)
        seismic_all = data['seismic']  
        n_traces_all = data['n_traces']  

        num_gathers = seismic_all.shape[0]
        if num_gathers > 10:
            return "Maksimal 10 gathers di .npz", 400

        predictions = []  
        vis_path = None  

        for idx in range(num_gathers):
            seismic = seismic_all[idx][:, :n_traces_all[idx]]  

            processed = preprocessor.preprocess(seismic)
            processed = np.repeat(processed, 3, axis=-1)
            input_model = np.expand_dims(processed, axis=0)
            pred = model.predict(input_model, verbose=0)[0]
            pred_prob = preprocessor.postprocess_mask(pred[..., 0])
            pred_binary = (pred_prob > 0.5).astype(np.float32)

            predictions.append((idx, pred_binary))

            if idx == 0:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(seismic, cmap='seismic', aspect='auto')
                axes[0].set_title(f"Input Seismic (Gather {idx})")
                axes[1].imshow(pred_prob, cmap='hot', aspect='auto')
                axes[1].set_title("Prediction Probability")
                axes[2].imshow(pred_binary, cmap='gray', aspect='auto')
                axes[2].set_title("Prediction Binary (Threshold 0.5)")
                plt.tight_layout()

                vis_filename = f'result_{filename}_gather0.png'
                vis_path = os.path.join('static/results', vis_filename)
                plt.savefig(vis_path)
                plt.close()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for idx, pred_binary in predictions:
                npz_data = {'pred_binary': pred_binary}  
                npz_bytes = io.BytesIO()
                np.savez(npz_bytes, **npz_data)
                npz_bytes.seek(0)
                zipf.writestr(f'pred_gather{idx}.npz', npz_bytes.read())

        zip_buffer.seek(0)
        zip_filename = f'predictions_{filename}.zip'

        return render_template('index.html',
                               uploaded_npz=filename,
                               vis_image=f'results/{vis_filename}',
                               zip_download=zip_filename)  

@app.route('/download_zip/<zip_filename>')
def download_zip(zip_filename):
    return "Download ZIP (implementasi lengkap nanti)"  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)