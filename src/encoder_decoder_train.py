import torch.nn as nn
import torch
from torchvision import datasets, transforms

device = torch.device('gpu')

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, lambda_1, lambda_2, lambda_tv):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.l2_loss = L2Loss()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_tv = lambda_tv

    def forward(self, input):
        input_encoded = self.encoder(input)['r41']
        output = self.decoder(input_encoded)
        output_encoded = self.encoder(output)['r41']
        loss, feature_loss, per_pixel_loss = self.calculate_loss(output, input, output_encoded, input_encoded)
        return output, loss, feature_loss, per_pixel_loss

    def calculate_loss(self, input, target, input_features, target_features):
        content_feature_loss = self.l2_loss(input_features.to(device), target_features.to(device))
        per_pixel_loss = self.l2_loss(input, target).to(device)
        tv_regularizer = self.tv_regularizer(input)
        loss = self.lambda_tv * tv_regularizer + \
               self.lambda_1 * content_feature_loss + \
               self.lambda_2 * per_pixel_loss
        return loss.to(device), content_feature_loss, per_pixel_loss

    def tv_regularizer(self, input, beta=2.):
        dy = torch.zeros(input.size())
        dx = torch.zeros(input.size())
        dy[:, 1:, :] = -input[:, :-1, :] + input[:, 1:, :]
        dx[:, :, 1:] = -input[:, :, :-1] + input[:, :, 1:]
        return torch.sum((dx.pow(2) + dy.pow(2)).pow(beta / 2.))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu_1_1 = nn.ReLU(inplace=True)

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)
        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)
        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_3 = nn.ReLU(inplace=True)
        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu_2_4 = nn.ReLU(inplace=True)
        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)
        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)
        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)
        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, input):
        # first block
        out = self.reflecPad_1_1(input)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # first block
        self.conv_1_1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu_1_2 = nn.ReLU(inplace=True)
        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_1_3 = nn.ReLU(inplace=True)
        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)
        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)
        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)
        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_3 = nn.ReLU(inplace=True)
        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_4 = nn.ReLU(inplace=True)
        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

    def forward(self, input):
        output = {}

        # first block
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)
        output['r11'] = out
        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)
        out = self.maxPool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        output['r21'] = out
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.maxPool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        output['r31'] = out
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)
        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)
        out = self.maxPool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        output['r41'] = out

        return output

def get_encoder_decoder_model(configuration):

    encoder_model_path = configuration['encoder_model_path']
    print('using the encoder from {}'.format(encoder_model_path))

    checkpoint = torch.load(encoder_model_path, map_location='cpu')
    print('loaded checkpoint')

    encoder = Encoder()
    print('got encoder model')
    encoder.load_state_dict(checkpoint)
    print('loaded encoder checkpoint')

    for param in encoder.parameters():
        param.requires_grad = False

    decoder = Decoder()
    print('got decoder model')

    # loss factors
    lambda_1 = configuration['lambda_1']
    lambda_2 = configuration['lambda_2']
    lambda_tv = configuration['lambda_tv']

    encoder_decoder_model = Autoencoder(encoder, decoder, lambda_1, lambda_2, lambda_tv)

    print('Let\'s use the {}'.format(device))

    print('printing whole model params which require_grad')
    for name, param in encoder_decoder_model.named_parameters():
        if param.requires_grad:
            print(name)

    return encoder_decoder_model


class L2Loss(nn.Module):
    """
    simple L_2-Loss
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean(torch.abs(input - target).pow(2)))

configuration = {
    'encoder_model_path' : "encoder_1_25_6_state_dict.pth",
    'lambda_1' : 25,
    'lambda_2' :1,
    'lambda_tv' : 0.000001,
}

model = get_encoder_decoder_model(configuration)

data_dir = './dataset'
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def training(model, train_loader, Epochs):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output_style, loss, content_feature_loss, per_pixel_loss = model(img)
            loss_curr = loss.sum()
            loss_curr.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss))
        
    return train_loss

optimizer = torch.optim.Adam(model.decoder.parameters(), 0.01)

train_loss = training(model, train_loader, 10)

torch.save(model, 'encoder_decoder.pth')