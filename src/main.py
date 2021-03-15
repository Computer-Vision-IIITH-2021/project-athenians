import os
import torch
import argparse
import time
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from WCT import WCT


def parseArgs():

    """Parse arguments passed to file during execution"""

    parser = argparse.ArgumentParser(description="WCT Pytorch")
    parser.add_argument("--contentPath", default="images/content", help="path to train")
    parser.add_argument("--stylePath", default="images/style", help="path to train")
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
    parser.add_argument("--outf", default="samples/", help="folder to output images")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="hyperparameter to blend wct feature and content feature",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="which gpu to run on.  default is 0"
    )

    args = parser.parse_args()
    return args


def styleTransfer(contentImg, styleImg, imgname, csF, wct):

    """Performs style transfer on content image and saves image"""

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

    vutils.save_image(Im.data.cpu().float(), os.path.join(args.outf, imgname))
    return


def run(args):

    """Calls style transfer function and measures average time taken"""

    wct = WCT(args)

    avgTime = 0
    cImg = torch.Tensor()
    sImg = torch.Tensor()
    csF = torch.Tensor()
    csF = Variable(csF)

    if args.cuda:
        cImg = cImg.cuda(args.gpu)
        sImg = sImg.cuda(args.gpu)
        csF = csF.cuda(args.gpu)
        wct.cuda(args.gpu)

    for i, (contentImg, styleImg, imgname) in enumerate(loader):
        imgname = imgname[0]
        print("Transferring " + imgname)

        if args.cuda:
            contentImg = contentImg.cuda(args.gpu)
            styleImg = styleImg.cuda(args.gpu)

        cImg = Variable(contentImg, volatile=True)
        sImg = Variable(styleImg, volatile=True)

        start_time = time.time()
        styleTransfer(cImg, sImg, imgname, csF, wct)
        end_time = time.time()

        print("Elapsed time is: %f" % (end_time - start_time))
        avgTime += end_time - start_time

    print("Processed %d images. Averaged time is %f" % ((i + 1), avgTime / (i + 1)))
    return


args = parseArgs()

try:
    os.makedirs(args.outf)
except OSError:
    pass

# Load dataset
dataset = Dataset(args.contentPath, args.stylePath, args.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

run(args)
