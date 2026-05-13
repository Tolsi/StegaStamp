import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    model = tf.saved_model.load(args.model, tags=['serve'])

    for filename in files_list:
        image = Image.open(filename).convert("RGB")
        image = ImageOps.fit(image, (400, 400))
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=0)
        
        result = model.signatures['serving_default'](
            image=tf.constant(image_np),
            secret=tf.zeros((1, 100), dtype=tf.float32)
        )
        
        secret = result['decoded'].numpy()[0]
        
        bits = ''.join(['1' if b > 0.5 else '0' for b in secret[:100]])
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        
        decoded = ''.join(chars).strip()
        print(f"{filename}: {decoded}")

if __name__ == "__main__":
    main()
