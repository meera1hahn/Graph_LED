import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from src.lingunet.language_model import RNN


def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LinearProjectionLayers(nn.Module):
    def __init__(
        self, image_channels, linear_hidden_size, rnn_hidden_size, num_hidden_layers
    ):
        super(LinearProjectionLayers, self).__init__()

        if num_hidden_layers == 0:
            # map pixel feature vector directly to score without activation
            self.out_layers = nn.Linear(image_channels + rnn_hidden_size, 1, bias=False)
        else:
            self.out_layers = nn.Sequential(
                nn.Conv2d(
                    image_channels + rnn_hidden_size,
                    linear_hidden_size,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.ReLU(),
                nn.Conv2d(linear_hidden_size, 1, kernel_size=1, padding=0, stride=1),
            )
            self.out_layers.apply(self.init_weights)

    def forward(self, x):
        return self.out_layers(x)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class LingUNet(nn.Module):
    def __init__(self, rnn_args, args):
        super(LingUNet, self).__init__()
        self.rnn_args = rnn_args

        self.m = args.num_lingunet_layers
        self.image_channels = args.linear_hidden_size
        self.freeze_resnet = args.freeze_resnet
        self.res_connect = args.res_connect
        self.device = args.device

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        if self.freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

        if not args.bidirectional:
            self.rnn_hidden_size = args.rnn_hidden_size
        else:
            self.rnn_hidden_size = args.rnn_hidden_size * 2
        assert self.rnn_hidden_size % self.m == 0

        self.rnn = RNN(
            rnn_args["input_size"],
            args.embed_size,
            args.rnn_hidden_size,
            args.num_rnn_layers,
            args.embed_dropout,
            args.bidirectional,
            args.embedding_dir,
        ).to(args.device)

        sliced_text_vector_size = self.rnn_hidden_size // self.m
        flattened_conv_filter_size = 1 * 1 * self.image_channels * self.image_channels
        self.text2convs = clones(
            nn.Linear(sliced_text_vector_size, flattened_conv_filter_size), self.m
        )

        self.conv_layers = nn.ModuleList([])
        for i in range(self.m):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.image_channels
                        if i == 0
                        else self.image_channels,
                        out_channels=self.image_channels,
                        kernel_size=5,
                        padding=2,
                        stride=1,
                    ),
                    nn.BatchNorm2d(self.image_channels),
                    nn.ReLU(True),
                )
            )

        # create deconv layers with appropriate paddings
        self.deconv_layers = nn.ModuleList([])
        for i in range(self.m):
            in_channels = self.image_channels if i == 0 else self.image_channels * 2
            out_channels = self.image_channels
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        padding=2,
                        stride=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )

        self.conv_dropout = nn.Dropout(p=0.25)
        self.deconv_dropout = nn.Dropout(p=0.25)

        self.out_layers = LinearProjectionLayers(
            image_channels=self.image_channels,
            linear_hidden_size=args.linear_hidden_size,
            rnn_hidden_size=0,
            num_hidden_layers=args.num_linear_hidden_layers,
        )
        self.sliced_size = self.rnn_hidden_size // self.m

        # initialize weights
        self.text2convs.apply(self.init_weights)
        self.conv_layers.apply(self.init_weights)
        self.deconv_layers.apply(self.init_weights)
        self.out_layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, images, texts, seq_lengths):
        B, num_maps, C, H, W = images.size()
        images = images.view(B * num_maps, C, H, W)
        images = self.resnet(images)

        batch_size, image_channels, height, width = images.size()

        text_embed = self.rnn(texts, seq_lengths)
        text_embed = torch.repeat_interleave(text_embed, num_maps, dim=0)

        Gs = []
        image_embeds = images

        for i in range(self.m):
            image_embeds = self.conv_dropout(image_embeds)
            image_embeds = self.conv_layers[i](image_embeds)
            text_slice = text_embed[
                :, i * self.sliced_size : (i + 1) * self.sliced_size
            ]

            conv_kernel_shape = (
                batch_size,
                self.image_channels,
                self.image_channels,
                1,
                1,
            )
            text_conv_filters = self.text2convs[i](text_slice).view(conv_kernel_shape)

            orig_size = image_embeds.size()
            image_embeds = image_embeds.view(1, -1, *image_embeds.size()[2:])
            text_conv_filters = text_conv_filters.view(
                -1, *text_conv_filters.size()[2:]
            )
            G = F.conv2d(image_embeds, text_conv_filters, groups=orig_size[0]).view(
                orig_size
            )
            image_embeds = image_embeds.view(orig_size)
            if self.res_connect:
                G = G + image_embeds
                G = F.relu(G)
            Gs.append(G)

        # deconvolution operations, from the bottom up
        H = Gs.pop()
        for i in range(self.m):
            if i == 0:
                H = self.deconv_dropout(H)
                H = self.deconv_layers[i](H)
            else:
                G = Gs.pop()
                concated = torch.cat((H, G), 1)
                H = self.deconv_layers[i](concated)

        out = self.out_layers(H).squeeze(-1)
        out = out.view(B, num_maps, out.size()[-2], out.size()[-1])
        out = F.log_softmax(out.view(B, -1), 1).view(B, num_maps, height, width)
        return out


def load_oldArgs(args, oldArgs):
    args.m = oldArgs["num_lingunet_layers"]
    args.image_channels = oldArgs["linear_hidden_size"]
    args.freeze_resnet = oldArgs["freeze_resnet"]
    args.res_connect = oldArgs["res_connect"]
    args.embed_size = oldArgs["embed_size"]
    args.rnn_hidden_size = oldArgs["rnn_hidden_size"]
    args.num_rnn_layers = oldArgs["num_rnn_layers"]
    args.embed_dropout = oldArgs["embed_dropout"]
    args.bidirectional = oldArgs["bidirectional"]
    args.linear_hidden_size = oldArgs["linear_hidden_size"]
    args.num_linear_hidden_layers = oldArgs["num_linear_hidden_layers"]
    args.ds_percent = oldArgs["ds_percent"]
    args.ds_height = oldArgs["ds_height"]
    args.ds_width = oldArgs["ds_width"]

    return args


def convert_model_to_state(model, args, rnn_args, cnn_args):
    state = {
        "args": vars(args),
        "rnn_args": rnn_args,
        "state_dict": {},
    }
    # use copies instead of references
    for k, v in model.state_dict().items():
        state["state_dict"][k] = v.clone().to(torch.device("cpu"))
    return state
