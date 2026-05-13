import glob
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret', type=str, default='Stega!!')
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    model = tf.saved_model.load(args.model, tags=['serve'])
    
    width = 400
    height = 400

    secret = []
    for c in args.secret[:12]:
        secret.extend([int(b) for b in format(ord(c), '08b')])
    while len(secret) < 100:
        secret.append(0)
    secret = np.array([secret], dtype=np.float32)

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = ImageOps.fit(image, size)
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_np = np.expand_dims(image_np, axis=0)
            
            result = model.signatures['serving_default'](
                image=tf.constant(image_np),
                secret=tf.constant(secret)
            )
            
            hidden_img = result['stegastamp'].numpy()
            residual = result['residual'].numpy()
            
            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            residual_vis = (residual[0] * 255).astype(np.uint8)
            
            save_name = os.path.basename(filename).split('.')[0]
            
            im = Image.fromarray(rescaled)
            im.save(args.save_dir + '/' + save_name + '_hidden.png')
            
            im = Image.fromarray(residual_vis)
            im.save(args.save_dir + '/' + save_name + '_residual.png')
            
            print(f"Saved to {args.save_dir}/{save_name}_hidden.png")

if __name__ == "__main__":
    main()
