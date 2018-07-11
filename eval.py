import os
import cv2
import string
from tqdm import tqdm
import click
import numpy as np
import editdistance
import glob

import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn


#model_path = './data/crnn.pth'
#img_path = '../TextBoxes_plusplus/docker_custom/crops/demo2/1.jpg'

@click.command()
@click.option('--image-path', type=str, default=None, help='Path to image')
@click.option('--alphabet', type=str, default="0123456789abcdefghijklmnopqrstuvwxyz", help='Alphabet to recognize')
@click.option('--snapshot', type=str, default="data/crnn.pth", help='Pre-trained weights')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')

def main(image_path, alphabet, snapshot, gpu, visualize):
    text = []
    if(os.path.isfile(image_path)):
        sim_pred = recognize(image_path, alphabet, snapshot, gpu)
        text.append(sim_pred)
    elif(os.path.isdir(image_path)):
        #loop over image directory
        print("is dir", image_path)
        file_list = []
        for file in sorted(os.listdir(image_path)):
            if file.endswith(".jpg"):
                file = os.path.join(image_path, file)
                sim_pred = recognize(file, alphabet, snapshot, gpu)
                text.append(sim_pred)

    if(len(text) > 0):
        print("recognized text: ", " ".join(text))


#bleh
def recognize(image_path, alphabet, snapshot, gpu):
    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % snapshot)
    model.load_state_dict(torch.load(snapshot))
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))

    image = Image.open(image_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

    return sim_pred

if __name__ == '__main__':
    main()
