from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import nn, Tensor
import copy

from sam2.build_sam import build_sam2
import tifffile as tiff
import torchvision.transforms.functional as func
from torchvision.transforms import InterpolationMode

from configs import config_2d
args = config_2d.args
vector_bins = args.vector_bins

backbone_channel_list = {
    "sam2_hiera_l" : [1152, 576, 288, 144],
    "sam2_hiera_s" : [768, 384, 192, 96]
}


class up_conv_2d(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(up_conv_2d, self).__init__()
		self.up = nn.Sequential(
			nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x

class down_conv_2d(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(down_conv_2d, self).__init__()
		self.down = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.down(x)
		return x


class res_conv_block_2d(nn.Module):
	"""
	Res Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(res_conv_block_2d, self).__init__()
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(out_ch))
		
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out += residual
		out = self.relu(out)
		return out


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False



# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size) 
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape
        out, hidden = self.lstm(x, hidden)

        if seq_len == 1:
            out1 = self.fc1(out[:, 0, :])
            out2 = self.fc1(out[:, 0, :])
            out3 = self.fc1(out[:, 0, :])
        else:
            out1 = self.fc1(out[:, 0, :])
            out2 = self.fc2(out[:, 1, :])
            out3 = self.fc3(out[:, 2, :])

        return [out1, out2, out3], hidden
	
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk, adadim) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, adadim),
            nn.GELU(),
            nn.Linear(adadim, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class SAM2_Net_2D(nn.Module):
	"""
	CSFL_Net_2D - Basic Implementation
	Paper: 
	"""

	def __init__(self, in_ch=1, out_ch=1, freeze_net = False, checkpoint_path=None, model_cfg=None, device=None):
		super(SAM2_Net_2D, self).__init__()

		if model_cfg is None: 
			model_cfg = "sam2_hiera_l.yaml"
		else:
			sam_types = ['sam2_hiera_l', 'sam2_hiera_s']
			assert model_cfg in sam_types
			model_cfg = model_cfg + '.yaml'
		
		if device is None:
			device = torch.device("cpu")

		if checkpoint_path:
			model = build_sam2(model_cfg, checkpoint_path, device=device)
		else:
			model = build_sam2(model_cfg)
		del model.sam_mask_decoder
		del model.sam_prompt_encoder
		del model.memory_encoder
		del model.memory_attention
		del model.mask_downsample
		del model.obj_ptr_tpos_proj
		del model.obj_ptr_proj
		del model.image_encoder.neck
		self.encoder = model.image_encoder.trunk

		for param in self.encoder.parameters():
			param.requires_grad = False
		blocks = []

		#* 加入Adapter
		for block in self.encoder.blocks:
			blocks.append(
				Adapter(block, args.adadim)
			)
		self.encoder.blocks = nn.Sequential(
			*blocks
		)
		self.backbone_channel_list = backbone_channel_list[model_cfg.split('.')[0]]
		ch_list = self.backbone_channel_list
		#* 统一channel数量
		self.rfb1 = RFB_modified(ch_list[3], args.rfbdim)
		self.rfb2 = RFB_modified(ch_list[2], args.rfbdim)
		self.rfb3 = RFB_modified(ch_list[1], args.rfbdim)
		self.rfb4 = RFB_modified(ch_list[0], args.rfbdim)
		self.up1 = (Up(args.rfbdim *2, args.rfbdim))
		self.up2 = (Up(args.rfbdim *2, args.rfbdim))
		self.up3 = (Up(args.rfbdim *2, args.rfbdim))
		self.up4 = (Up(args.rfbdim *2, args.rfbdim))
		self.side1 = nn.Conv2d(args.rfbdim, 1, kernel_size=1)
		self.side2 = nn.Conv2d(args.rfbdim, 1, kernel_size=1)
		self.head = nn.Conv2d(args.rfbdim, 1, kernel_size=1)
		self.chead = nn.Conv2d(args.rfbdim, 1, kernel_size=1)

		self.rnn_trans0 = nn.Linear(in_features=11*11*args.rfbdim//4, out_features=1024)
		self.rnn_trans1 = nn.Linear(in_features=11*11*args.rfbdim//4, out_features=1024)


		######################################*
		self.n1 = 352
		filters = [self.n1, self.n1 * 2, self.n1 * 4, self.n1 * 8, self.n1 * 16]
		
		# RNN parameter
		self.layer_num_RNN = 1
		self.seq_len = 3

		self.input_size_RNN = 128
		self.hidden_size_RNN = 100 
		self.output_size_direction = vector_bins #50
		self.output_size_radius = 1


		# self.conv_input = nn.Sequential(nn.Conv2d(in_ch, filters[0], kernel_size=3, stride=1, padding=1, bias=True), 
		# 								nn.BatchNorm2d(filters[0]),
		# 								nn.ReLU(inplace=True))

		# self.Conv1 = res_conv_block_2d(filters[0], filters[0])
		# self.Down1 = down_conv_2d(filters[0], filters[1])

		# self.Conv2 = res_conv_block_2d(filters[1], filters[1])
		# self.Down2 = down_conv_2d(filters[1], filters[2])

		# self.Conv3 = res_conv_block_2d(filters[2], filters[2])
		# self.Down3 = down_conv_2d(filters[2], filters[3])

		# self.Conv4 = res_conv_block_2d(filters[3], filters[3])
		# self.Down4 = down_conv_2d(filters[3], filters[4])

		# self.Conv5_1 = res_conv_block_2d(filters[4], filters[4])
		# self.Conv5_2 = res_conv_block_2d(filters[4], filters[4])
		# self.Conv5_3 = res_conv_block_2d(filters[4], filters[4])

		# self.Up5 = up_conv_2d(filters[4], filters[3])
		# self.Up_conv5 = res_conv_block_2d(filters[3], filters[3])

		# self.Up4 = up_conv_2d(filters[3], filters[2])
		# self.Up_conv4 = res_conv_block_2d(filters[2], filters[2])

		# self.Up3 = up_conv_2d(filters[2], filters[1])
		# self.Up_conv3 = res_conv_block_2d(filters[1], filters[1])

		# self.Up2 = up_conv_2d(filters[1], filters[0])
		# self.Up_conv2 = res_conv_block_2d(filters[0], filters[0])
		
		# # seg block
		# self.Conv6_1 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
		# 								nn.BatchNorm2d(filters[0]),
		# 								nn.ReLU(inplace=True))
		# self.Conv6_2 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
		# 								nn.BatchNorm2d(filters[0]),
		# 								nn.ReLU(inplace=True))
		# self.Conv6_out = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1),
		# 								nn.Sigmoid())

		# # centerline block
		# self.Conv7_1 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
		# 								nn.BatchNorm2d(filters[0]),
		# 								nn.ReLU(inplace=True))
		# self.Conv7_2 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
		# 								nn.BatchNorm2d(filters[0]),
		# 								nn.ReLU(inplace=True))
		# self.Conv7_out = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1),
		# 								nn.Sigmoid())
		if freeze_net:
			print("freezing")
			freeze(self)			
		
		# direction head
		filters[0] = int(args.rfbdim // 4)
		self.Tracer1 = nn.Sequential(nn.Conv2d(args.rfbdim, filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm2d(filters[0]),
										nn.ReLU(inplace=True))
		self.Tracer2 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm2d(filters[0]),
										nn.ReLU(inplace=True))
		self.Tracer3 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True),
										nn.BatchNorm2d(filters[0]),
										nn.ReLU(inplace=True))

		self.Tracer4_RNN_1 = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_direction, self.seq_len)
		self.Tracer4_RNN_2 = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_direction, self.seq_len)
  
		# radius head
		self.Radius1 = nn.Sequential(nn.Conv2d(args.rfbdim, filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm2d(filters[0]),
										nn.ReLU(inplace=True))
		self.Radius2 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True), 
										nn.BatchNorm2d(filters[0]),
										nn.ReLU(inplace=True))
		self.Radius3 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1,  padding=1, bias=True),
										nn.BatchNorm2d(filters[0]),
										nn.ReLU(inplace=True))
		self.Radius4_RNN = LSTM(self.input_size_RNN, self.hidden_size_RNN, self.output_size_radius, self.seq_len)
		# RNN Block


	def forward(self, x_, mode = 'train'):
		outputs_img = []
		hidden_direction_radius = []
		batch_size, seq_len, c, h, w = x_.size()
		
		if mode == 'train':
			for t in range(seq_len): 
				input_image = x_[:, t, :, :, :]

				x1, x2, x3, x4 = self.encoder(input_image)
				x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
				x = self.up1(x4, x3)
				out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
				x = self.up2(x, x2)
				out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
				x = self.up3(x, x1)
				d_seg = F.interpolate(self.head(x), scale_factor=4, mode='bilinear') 
				d_seg = nn.Sigmoid()(d_seg)

				d_centerline = F.interpolate(self.chead(x), scale_factor=4, mode='bilinear')
				d_centerline = nn.Sigmoid()(d_centerline)

				e5 = x4
				# d0 = self.conv_input(input_image)
				# e1 = self.Conv1(d0)
				# e1_d = self.Down1(e1)

				# e2 = self.Conv2(e1_d)
				# e2_d = self.Down2(e2)

				# e3 = self.Conv3(e2_d)
				# e3_d = self.Down3(e3)

				# e4 = self.Conv4(e3_d)
				# e4_d = self.Down4(e4)

				# e5_2 = self.Conv5_1(e4_d)
				# e5_1 = self.Conv5_2(e5_2)
				# e5 = self.Conv5_3(e5_1)

				# d5 = self.Up5(e5)
				# d5 = torch.add(e4, d5)
				# d5 = self.Up_conv5(d5)

				# d4 = self.Up4(d5)
				# d4 = torch.add(e3, d4)
				# d4 = self.Up_conv4(d4)

				# d3 = self.Up3(d4)
				# d3 = torch.add(e2, d3)
				# d3 = self.Up_conv3(d3)

				# d2 = self.Up2(d3)
				# d2 = torch.add(e1, d2)
				# d2 = self.Up_conv2(d2)


				# # exist block
				# d6_1 = self.Conv6_1(d2)	
				# d6_2 = self.Conv6_2(d6_1)	
				# d_seg = self.Conv6_out(d6_2)

				# # centerline block
				# d7_1 = self.Conv7_1(d2)	
				# d7_2 = self.Conv7_2(d7_1)	
				# d_centerline = self.Conv7_out(d7_2)	

				# direction block
				t1 = self.Tracer1(e5)
				t2 = self.Tracer2(t1)
				t3 = self.Tracer3(t2)
				dim = t3.shape[1] * t3.shape[2] * t3.shape[3]
				t3_flatten = t3.reshape(-1, dim)

				# radius block
				r1 = self.Radius1(e5)
				r2 = self.Radius2(r1)
				r3 = self.Radius3(r2)
				dim = r3.shape[1] * r3.shape[2] * r3.shape[3]
				r3_flatten = r3.reshape(-1, dim)


				outputs_img += [[d_seg, d_centerline]]
				hidden_direction_radius += [[t3_flatten, r3_flatten]]


			# 输出 image output
			output_seg = outputs_img[0][0].unsqueeze(1)
			output_centerline = outputs_img[0][1].unsqueeze(1)
			for t in range(1, seq_len): 
				output_seg_temp = outputs_img[t][0].unsqueeze(1)
				output_centerline_temp = outputs_img[t][1].unsqueeze(1)

				output_seg = torch.cat([output_seg, output_seg_temp], dim=1)
				output_centerline = torch.cat([output_centerline, output_centerline_temp], dim=1)


			# 输出 direction/radius multi output
			input_t3 = hidden_direction_radius[0][0].unsqueeze(1)
			input_r3 = hidden_direction_radius[0][1].unsqueeze(1)
			for t in range(1, seq_len): 
			
				input_t3_temp = hidden_direction_radius[t][0].unsqueeze(1)
				input_r3_temp = hidden_direction_radius[t][1].unsqueeze(1)

				input_t3 = torch.cat([input_t3, input_t3_temp], dim=1)
				input_r3 = torch.cat([input_r3, input_r3_temp], dim=1)

   
			# 计算RNN模块
			hidden_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			hidden_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')

			hidden_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')


			# hidden_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN)
			# cell_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN)
			# hidden_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN)
			# cell_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN)

			# hidden_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN)
			# cell_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN)
			#! 精简维度
			# input_t3 = self.rnn_trans0(input_t3)
			# input_t3 = nn.ReLU()(input_t3)

			# input_r3 = self.rnn_trans1(input_r3)
			# input_r3 = nn.ReLU()(input_r3)

			[output1_d1, output2_d1, output3_d1], (hidden_d1, cell_d1) = self.Tracer4_RNN_1(input_t3, (hidden_d1, cell_d1))
			[output1_d2, output2_d2, output3_d2], (hidden_d2, cell_d2) = self.Tracer4_RNN_2(input_t3, (hidden_d2, cell_d2))
			[output1_r, output2_r, output3_r] , (hidden_r, cell_r) = self.Radius4_RNN(input_r3, (hidden_r, cell_r))


			output_d1 = torch.cat([output1_d1.unsqueeze(1), output2_d1.unsqueeze(1), output3_d1.unsqueeze(1)], dim=1)
			output_d2 = torch.cat([output1_d2.unsqueeze(1), output2_d2.unsqueeze(1), output3_d2.unsqueeze(1)], dim=1)
			output_r = torch.cat([output1_r, output2_r, output3_r], dim=1)

   
			return output_seg, output_centerline, output_d1, output_d2, output_r

		elif mode == 'test_dis':
			input_image = x_[:, 0, :, :, :]

			x1, x2, x3, x4 = self.encoder(input_image)
			x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
			x = self.up1(x4, x3)
			out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
			x = self.up2(x, x2)
			out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
			x = self.up3(x, x1)

			d_seg = F.interpolate(self.head(x), scale_factor=4, mode='bilinear') 
			d_seg = nn.Sigmoid()(d_seg)
			
			d_centerline = F.interpolate(self.chead(x), scale_factor=4, mode='bilinear')
			d_centerline = nn.Sigmoid()(d_centerline)

			# d0 = self.conv_input(input_image)

			# e1 = self.Conv1(d0)
			# e1_d = self.Down1(e1)

			# e2 = self.Conv2(e1_d)
			# e2_d = self.Down2(e2)

			# e3 = self.Conv3(e2_d)
			# e3_d = self.Down3(e3)

			# e4 = self.Conv4(e3_d)
			# e4_d = self.Down4(e4)

			# e5_2 = self.Conv5_1(e4_d)
			# e5_1 = self.Conv5_2(e5_2)
			# e5 = self.Conv5_3(e5_1)
				

			# d5 = self.Up5(e5)
			# d5 = torch.add(e4, d5)
			# d5 = self.Up_conv5(d5)

			# d4 = self.Up4(d5)
			# d4 = torch.add(e3, d4)
			# d4 = self.Up_conv4(d4)

			# d3 = self.Up3(d4)
			# d3 = torch.add(e2, d3)
			# d3 = self.Up_conv3(d3)

			# d2 = self.Up2(d3)
			# d2 = torch.add(e1, d2)
			# d2 = self.Up_conv2(d2)


			# # exist block
			# d6_1 = self.Conv6_1(d2)	
			# d6_2 = self.Conv6_2(d6_1)	
			# d_seg = self.Conv6_out(d6_2)

			# # centerline block
			# d7_1 = self.Conv7_1(d2)	
			# d7_2 = self.Conv7_2(d7_1)	
			# d_centerline = self.Conv7_out(d7_2)	
			return d_seg, d_centerline
		
		elif mode == 'test_d':
			for t in range(seq_len): 
				input_image = x_[:, t, :, :, :]

				x1, x2, x3, x4 = self.encoder(input_image)
				x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
				x = self.up1(x4, x3)
				out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
				x = self.up2(x, x2)
				out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
				x = self.up3(x, x1)
				# d_seg = F.interpolate(self.head(x), scale_factor=4, mode='bilinear') 
				# d_seg = nn.Sigmoid()(d_seg)

				# d_centerline = F.interpolate(self.chead(x), scale_factor=4, mode='bilinear')
				# d_centerline = nn.Sigmoid()(d_centerline)

				e5 = x4

				# direction block
				t1 = self.Tracer1(e5)
				t2 = self.Tracer2(t1)
				t3 = self.Tracer3(t2)
				dim = t3.shape[1] * t3.shape[2] * t3.shape[3]
				t3_flatten = t3.reshape(-1, dim)

				# radius block
				r1 = self.Radius1(e5)
				r2 = self.Radius2(r1)
				r3 = self.Radius3(r2)
				dim = r3.shape[1] * r3.shape[2] * r3.shape[3]
				r3_flatten = r3.reshape(-1, dim)


				#outputs_img += [[d_seg, d_centerline]]
				hidden_direction_radius += [[t3_flatten, r3_flatten]]
			
			# 输出 direction/radius multi output
			input_t3 = hidden_direction_radius[0][0].unsqueeze(1)
			input_r3 = hidden_direction_radius[0][1].unsqueeze(1)
			for t in range(1, seq_len): 
				input_t3_temp = hidden_direction_radius[t][0].unsqueeze(1)
				input_r3_temp = hidden_direction_radius[t][1].unsqueeze(1)

				input_t3 = torch.cat([input_t3, input_t3_temp], dim=1)
				input_r3 = torch.cat([input_r3, input_r3_temp], dim=1)

			# 计算RNN模块
			hidden_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d1 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			hidden_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_d2 = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')

			hidden_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')
			cell_r = torch.zeros(self.layer_num_RNN, batch_size, self.hidden_size_RNN).to('cuda')


			[output1_d1, output2_d1, output3_d1], (hidden_d1, cell_d1) = self.Tracer4_RNN_1(input_t3, (hidden_d1, cell_d1))
			[output1_d2, output2_d2, output3_d2], (hidden_d2, cell_d2) = self.Tracer4_RNN_2(input_t3, (hidden_d2, cell_d2))

			[output1_r, output2_r, output3_r] , (hidden_r, cell_r) = self.Radius4_RNN(input_r3, (hidden_r, cell_r))


			output_d1 = torch.cat([output1_d1.unsqueeze(1), output2_d1.unsqueeze(1), output3_d1.unsqueeze(1)], dim=1)
			output_d2 = torch.cat([output1_d2.unsqueeze(1), output2_d2.unsqueeze(1), output3_d2.unsqueeze(1)], dim=1)
			output_r = torch.cat([output1_r, output2_r, output3_r], dim=1)

			
			return output_d1, output_d2, output_r
		else:
			print("choose a training mode")


if __name__ == "__main__":
	trial_img = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/dataset/drive/training_datasets/21/1_pos_0/node_img.tif"
	pretrain_path = "/mnt/40B2A1DBB2A1D5A6/lyx/project/MODEL/Sam2/sam2_hiera_large.pt"
	device = torch.device("cuda:0")
	img = tiff.imread(trial_img)

	model = SAM2_Net_2D(1, 1, checkpoint_path=pretrain_path, device=device)
	model = model.to(device)
	
	img = torch.tensor(img)
	img = torch.permute(img, [0,3,1,2])
	img = func.resize(img, (352,352), interpolation=InterpolationMode.BICUBIC)
	img = torch.reshape(img, (-1,3,3,352,352))
	img = img / 255
	img = img.to(device)



	_ = model(img)

	print()
