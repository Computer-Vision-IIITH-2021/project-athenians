import os
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, content_Img_path, style_Img_path, size_Desired):
        super(Dataset, self).__init__()
        self.prep = transforms.Compose(
            [transforms.Scale(size_Desired), transforms.ToTensor()]
        )
        self.content_Img_path = content_Img_path
        self.style_Img_path = style_Img_path

        self.size_Desired = size_Desired

        image_list = []
        for img in os.listdir(content_Img_path):
            img_lis = img.split(".")
            extensions = ["png", "jpg", "jpeg"]
            if img_lis[1] in extensions:
                image_list.append(img)
        self.image_list = image_list

    def __getitem__(self, index):
        content_img = Image.open(
            os.path.join(self.content_Img_path, self.image_list[index])
        ).convert("RGB")
        style_img = Image.open(
            os.path.join(self.style_Img_path, self.image_list[index])
        ).convert("RGB")

        if self.size_Desired:
            width, height = content_img.size
            if width > height:
                if width != self.size_Desired:
                    content_img = content_img.resize(
                        (self.size_Desired, int(height * self.size_Desired / width))
                    )
                    style_img = style_img.resize(
                        (self.size_Desired, int(height * self.size_Desired / width))
                    )

            else:
                if height != self.size_Desired:
                    self.size_Desired = int(width * self.size_Desired / height)
                    content_img = content_img.resize(
                        (self.size_Desired, self.size_Desired)
                    )
                    style_img = style_img.resize((self.size_Desired, self.size_Desired))

        content_img = transforms.ToTensor()(content_img)
        style_img = transforms.ToTensor()(style_img)
        content_img_squeezed = content_img.squeeze(0)
        style_img_squeezed = style_img.squeeze(0)
        cur_img = self.image_list[index]

        return content_img_squeezed, style_img_squeezed, cur_img

    def __len__(self):
        return len(self.image_list)
