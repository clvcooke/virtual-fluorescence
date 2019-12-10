from torch import nn


class Classifier(nn.Module):

    def __init__(self, num_classes, channels_in=1, batch_norm=False):
        super().__init__()
        # we will follow a super simple classiier structure
        # input image is going to be 28x28xc where c is the number of channels
        start_filters = 4
        self.conv1 = self.gen_conv_block(channels_in, start_filters, kernel_size=3, batch_norm=batch_norm)
        self.conv2 = self.gen_conv_block(start_filters, start_filters * 2, kernel_size=3, batch_norm=batch_norm)
        self.conv3 = self.gen_conv_block(start_filters * 2, start_filters * 4, kernel_size=3, batch_norm=batch_norm)
        dims = 3
        # flatten
        self.flat_shape = int(dims * dims * start_filters * 4)
        self.dense = nn.Sequential(
            nn.Linear(self.flat_shape, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            nn.Softmax()
        )

    @staticmethod
    def gen_conv_block(channels_in, channels_out, kernel_size=3,
                       pooling=True, batch_norm=False):
        layers = []
        layers.append(nn.Conv2d(channels_in, channels_out, kernel_size, padding=[1, 1]))
        if batch_norm:
            layers.append(nn.BatchNorm2d(channels_out))
        layers += [nn.ReLU(),
                   nn.Conv2d(channels_out, channels_out, kernel_size, padding=[1, 1])]
        if batch_norm:
            layers.append(nn.BatchNorm2d(channels_out))
        layers.append(nn.ReLU())
        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.dense(x3.view(-1, self.flat_shape))
        return x4
