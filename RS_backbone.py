from mmdet.models.backbones import DetectoRS_ResNet


def test_detectorrs_resnet_backbone():
    detectorrs_cfg = dict(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True)
    """Test init_weights config"""
    model = DetectoRS_ResNet(**detectorrs_cfg)
    # print(model)
    print(model.conv1.weight.shape)


if __name__ == '__main__':
    test_detectorrs_resnet_backbone()
"""
DetectoRS_ResNet(
  (conv1): ConvAWS2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): ResLayer(
    (0): Bottleneck(
      (conv1): ConvAWS2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ConvAWS2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): ConvAWS2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): ConvAWS2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ConvAWS2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): ConvAWS2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ConvAWS2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer2): ResLayer(
    (0): Bottleneck(
      (conv1): ConvAWS2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (switch): Conv2d(128, 1, kernel_size=(1, 1), stride=(2, 2))
        (pre_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(128, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (offset_l): Conv2d(128, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): ConvAWS2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): ConvAWS2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): ConvAWS2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): ConvAWS2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer3): ResLayer(
    (0): Bottleneck(
      (conv1): ConvAWS2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (switch): Conv2d(256, 1, kernel_size=(1, 1), stride=(2, 2))
        (pre_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(256, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (offset_l): Conv2d(256, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): ConvAWS2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): ConvAWS2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): ConvAWS2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): ConvAWS2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): Bottleneck(
      (conv1): ConvAWS2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): Bottleneck(
      (conv1): ConvAWS2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer4): ResLayer(
    (0): Bottleneck(
      (conv1): ConvAWS2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (switch): Conv2d(512, 1, kernel_size=(1, 1), stride=(2, 2))
        (pre_context): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(512, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (offset_l): Conv2d(512, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): ConvAWS2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): ConvAWS2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): ConvAWS2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): SAConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (switch): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        (pre_context): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (post_context): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (offset_s): Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (offset_l): Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): ConvAWS2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
)
"""
