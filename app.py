import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("landColorGenV1.keras")

def generate_image(input_img):
    input_img = tf.convert_to_tensor(input_img)
    input_img = tf.cast(input_img,tf.float32)
    init_shape = input_img.shape
    input_img = tf.image.resize(input_img, [256, 256],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_img = (input_img / 127.5) -1
    input_img = tf.reshape(input_img,(1,256,256,3))
    output = model(input_img,training=True)
    # out_img =  output[0].numpy()* 0.5 + 0.5
    out_img = tf.image.resize(output[0], [init_shape[0],init_shape[1]],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    out_img = out_img.numpy()*0.5 + 0.5
    return out_img
app =  gr.Interface(fn = generate_image, inputs="image", outputs="image")
app.launch(debug=False)