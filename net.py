import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# define style loss and content loss
# both losses calculate MSE between two sets of feature maps or gram matrices:
#     one coming from the input image, another - from the content or the style image, accordingly

# to compute style loss, we will need a gram matrix as a normalized result of multiplying a matrix
#     by its transposed matrix
# here, the original matrix is a reshaped version of the feature maps of a layer,
#     with dimensions corresponding to the number of feature maps in a layer and their lengths


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# we will use VGG19. It is trained on normalized images, so we need to niormalize our images as well
# we will build a normalization class to fit it into nn.Sequential


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# function to resize the image, transform it to tensor, and add a fake batch dimension with unsqueeze
def load_image(image_name):

    #imsize = 256 if torch.cuda.is_available() else 128
    imsize = 256

    loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# we will build a new nn.Sequential by putting Normalization in the beginning,
#     iterating over the layers of the initial imported model, and trimming


class Net():
    def __init__(self):
        # import pretrained VGG19 network
        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_model(self, style_img, content_img):
        # normalization
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(device)

        content_losses = []
        style_losses = []

        content_img = load_image(content_img)
        style_img = load_image(style_img)

        # iterate through layers
        model = nn.Sequential(normalization)

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)

            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            # add content loss
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            # add style loss
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses, content_img

    # in this application, we need to optimize not the model parameters, but the input image itself

    def run_style_transfer(self, model, style_losses, content_losses, input_img
                           , num_steps=300, style_weight=1000000, content_weight=1):

        # update requires_grad fields
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = optim.LBFGS([input_img])

        run = [0]
        while run[0] <= num_steps:

            def closure():

                # correct the values of updated input (content) image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
