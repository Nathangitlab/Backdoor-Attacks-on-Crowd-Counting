import json
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    img_folder = './dataset/part_B_final/train_data/High_Density_Rain2_BG_portion0.2_images_train_data'

    # path to the final json file
    output_json = './part_B_train_Rain2_portion0.2.json'

    img_list = []

    for img_path in glob.glob(join(img_folder,'*.jpg')):
        img_list.append(img_path)

    with open(output_json,'w') as f:
        json.dump(img_list,f)
