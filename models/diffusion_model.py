import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
import numpy as np

class DiffusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet', dataset_mode='tuning')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['f']
        self.model_names = ['netf']
        self.visual_names = ['lr', 'hr']

        # network
        self.netf = networks.define_F(opt).to(self.device)
        self.ema = networks.ExponentialMovingAverage(self.netf.parameters(), decay=0.999)
        self.optimizer_f = torch.optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_f)

        self.sigmas = [torch.tensor(opt.sigma).float()]
        self.acc = 0  # training iteration index
        self.sigma = self.sigmas[0]  # default initialization

    def set_input(self, input):
        self.hr = input['B'].to(self.device)
        self.lr = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.zeros = torch.zeros(self.sigma.shape).to(self.device, dtype=torch.float32)
        self.noises = torch.randn_like(self.hr) * self.sigma
        self.xt = self.hr + self.noises
        self.zt = self.xt - self.noises
        self.cond = self.xt - self.zt

    def backward_f(self):
        _, self.loss_f = self.netf(self.hr, self.sigma)
        self.loss_f.backward()

    def optimize_parameters(self):
        self.set_sigma(self.acc)
        self.optimizer_f.zero_grad()
        self.backward_f()
        torch.nn.utils.clip_grad_norm_(self.netf.parameters(), 1)
        self.optimizer_f.step()
        self.ema.update(self.netf.parameters())
        with torch.no_grad():
            self.forward()

    def set_sigma(self, global_step):
        self.sigma = self.sigmas[0].to(self.device)
