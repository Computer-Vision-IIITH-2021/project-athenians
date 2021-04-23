import os
from skimage.draw.draw import circle
import torch
import argparse
import time
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from WCT import WCT
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2


torch.backends.cudnn.benchmark = True


def parseArgs():

    """Parse arguments passed to file during execution"""

    parser = argparse.ArgumentParser(description="WCT Pytorch")
    parser.add_argument("--contentPath", default="images/content", help="path to train")
    parser.add_argument("--stylePath", default="images/style", help="path to train")
    parser.add_argument(
        "--outf", default="images/samples", help="folder to output images"
    )
    parser.add_argument(
        "--scaleStyle",
        type=int,
        default=512,
        help="scale style image: size of style image",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="hyperparameter to blend wct feature and content feature",
    )
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--vgg1",
        default="models/vgg_normalised_conv1_1.t7",
        help="Path to the VGG conv1_1",
    )
    parser.add_argument(
        "--vgg2",
        default="models/vgg_normalised_conv2_1.t7",
        help="Path to the VGG conv2_1",
    )
    parser.add_argument(
        "--vgg3",
        default="models/vgg_normalised_conv3_1.t7",
        help="Path to the VGG conv3_1",
    )
    parser.add_argument(
        "--vgg4",
        default="models/vgg_normalised_conv4_1.t7",
        help="Path to the VGG conv4_1",
    )
    parser.add_argument(
        "--vgg5",
        default="models/vgg_normalised_conv5_1.t7",
        help="Path to the VGG conv5_1",
    )
    parser.add_argument(
        "--decoder5",
        default="models/feature_invertor_conv5_1.t7",
        help="Path to the decoder5",
    )
    parser.add_argument(
        "--decoder4",
        default="models/feature_invertor_conv4_1.t7",
        help="Path to the decoder4",
    )
    parser.add_argument(
        "--decoder3",
        default="models/feature_invertor_conv3_1.t7",
        help="Path to the decoder3",
    )
    parser.add_argument(
        "--decoder2",
        default="models/feature_invertor_conv2_1.t7",
        help="Path to the decoder2",
    )
    parser.add_argument(
        "--decoder1",
        default="models/feature_invertor_conv1_1.t7",
        help="Path to the decoder1",
    )
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--fineSize",
        type=int,
        default=512,
        help="resize image to fineSize x fineSize,leave it to 0 if not resize",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="which gpu to run on.  default is 0"
    )

    parser.add_argument(
        "--masksContentPath",
        default="mask_images/content.jpg",
        help="path to content image",
    )
    parser.add_argument(
        "--masksStylePath",
        default="mask_images/style/",
        help="path to style images directory",
    )

    parser.add_argument(
        "--masksResultPath",
        default="mask_images/result.jpg",
        help="path to result for masks",
    )

    args = parser.parse_args()
    return args


def performStyleTransfer(args, contentImg, styleImg, csF, wct):

    """Performs style transfer on content image and returns transferred image"""

    enc = [wct.e5, wct.e4, wct.e3, wct.e2, wct.e1]
    dec = [wct.d5, wct.d4, wct.d3, wct.d2, wct.d1]

    Im = contentImg

    for i in range(len(enc)):
        sF = enc[i](styleImg)
        cF = enc[i](Im)
        sF = sF.data.cpu().squeeze(0)
        cF = cF.data.cpu().squeeze(0)
        csF = wct.transform(cF, sF, csF, args.alpha)
        Im = dec[i](csF)

    return Im.data.cpu().float().squeeze(0)


def style_transfer_multiple_styles(args, contentImg, styleImgList):
    """Applies style transfer to the content image with the corresponding style images and returns a list of transfer"""

    num_style_images = len(styleImgList)
    style_transferred_images = []

    wct = WCT(args)
    csF = torch.Tensor()
    csF = Variable(csF)
    if args.cuda:
        wct = wct.cuda(args.gpu)
        csF = csF.cuda(args.gpu)

    for i in range(num_style_images):
        cImg = Variable(contentImg, volatile=True)
        sImg = Variable(styleImgList[i], volatile=True)

        transferred_img = performStyleTransfer(args, cImg, sImg, csF, wct)
        style_transferred_images.append(transferred_img)

    return style_transferred_images


def run_style_transfer_with_masks(args):
    """Runs style transfer with masks"""

    content_image_path = args.masksContentPath
    style_images_directory_path = args.masksStylePath

    content_img = Image.open(content_image_path).convert("RGB")

    width, height = content_img.size

    # Generate mask
    mask = np.zeros((height, width))
    rr, cc = circle(int(height / 2), int(width / 2), 600)
    # mask[0 : int(height / 2), :] = 1
    mask[rr, cc] = 1


    style_image_names_list = []
    for img in os.listdir(style_images_directory_path):
        img_lis = img.split(".")
        extensions = ["png", "jpg", "jpeg"]
        if img_lis[1] in extensions:
            style_image_names_list.append(img)
            print("Using image: ", img)

    size_Desired = args.fineSize
    scaleStyle = args.scaleStyle

    style_image_list = []

    if width > height:
        if width != size_Desired:
            content_img = content_img.resize(
                (size_Desired, int(height * size_Desired / width))
            )
            mask = cv2.resize(mask, content_img.size, interpolation=cv2.INTER_NEAREST)
            for img_name in style_image_names_list:
                style_img = Image.open(
                    os.path.join(style_images_directory_path, img_name)
                ).convert("RGB")
                style_img = style_img.resize(
                    (scaleStyle, int(height * scaleStyle / width))
                )
                style_img = transforms.ToTensor()(style_img)
                style_img = style_img.squeeze(0)
                style_img = style_img.unsqueeze(0)

                if args.cuda:
                    style_img = style_img.cuda(args.gpu)

                style_image_list.append(style_img)
    else:
        if height != size_Desired:
            size_Desired = int(width * size_Desired / height)
            content_img = content_img.resize((size_Desired, size_Desired))
            mask = cv2.resize(mask, content_img.size, interpolation=cv2.INTER_NEAREST)

            for img_name in style_image_names_list:
                style_img = Image.open(
                    os.path.join(style_images_directory_path, img_name)
                ).convert("RGB")
                style_img = style_img.resize((scaleStyle, scaleStyle))
                style_img = transforms.ToTensor()(style_img)
                style_img = style_img.squeeze(0)
                style_img = style_img.unsqueeze(0)

                if args.cuda:
                    style_img = style_img.cuda(args.gpu)

                style_image_list.append(style_img)

    content_img = transforms.ToTensor()(content_img)
    content_img = content_img.squeeze(0)
    content_img = content_img.unsqueeze(0)

    if args.cuda:
        content_img = content_img.cuda(args.gpu)

    # Get style transferred images
    style_transferred_images = style_transfer_multiple_styles(
        args, content_img, style_image_list
    )

    result_image = Image.new(
        "RGB",
        (style_transferred_images[0].shape[1], style_transferred_images[0].shape[2]),
    )
    result_image = np.array(result_image, dtype="uint8")

    mask = mask.astype("uint8")

    for i in range(len(style_transferred_images)):
        image_new = style_transferred_images[i]
        vutils.save_image(image_new, "tmpimg.jpg")
        image_new = cv2.imread("tmpimg.jpg")
        os.remove("tmpimg.jpg")
        result_image[mask == i] = (image_new)[mask == i]

    cv2.imwrite(args.masksResultPath, result_image)
    return


args = parseArgs()

try:
    os.makedirs(args.outf)
except OSError:
    pass

run_style_transfer_with_masks(args)
