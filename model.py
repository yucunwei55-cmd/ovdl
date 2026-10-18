"""CinC2017 V2 骨干：保留参数名称、初始化顺序及前向行为。

仅保留实际使用的 SE 模块和 ResNet18；移除未参与 forward 的 sn2。
注意原 V2 调用 ChannelAttention1d(planes, planes//16)，实际隐藏宽度
一直为16。这里刻意保留该行为，不能擅自改成标准 SE reduction=16。
"""
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import base, neuron, surrogate
from online_trace import TraceConv1d, TraceMemory

def SpikeNode(tau=2.0, v_threshold=0.5):
    return neuron.LIFNode(tau=tau, decay_input=True, v_threshold=v_threshold,
                          v_reset=0.0, detach_reset=True,
                          surrogate_function=surrogate.Sigmoid(alpha=4.0),
                          step_mode='s', backend='torch')

def make_trace_conv(config, *args, **kwargs):
    if config['method'] == 'current':
        return nn.Conv1d(*args, **kwargs)
    ratio = 1 if config['method'] == 'ottt' else config['trace_ratio']
    return TraceConv1d(*args, ratio=ratio, decay=config['trace_decay'],
                       tile=config['trace_tile'],
                       implementation=config['trace_backward'], **kwargs)

class SpikeWrapper(nn.Module):
    def __init__(self, node: nn.Module):
        super().__init__()
        self.node = node

    def forward(self, x):
        y = self.node(x)
        if isinstance(y, (list, tuple)):
            return y[0]
        return y

class ChannelAttention1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)


        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()

        # 平均池化路径
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1)

        # 最大池化路径
        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc(y_max).view(b, c, 1)

        # 结合两种注意力
        y = y_avg + y_max
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_attention='se', trace_config=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sn1 = SpikeWrapper(SpikeNode(tau=2.0, v_threshold=0.5))
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = make_trace_conv(trace_config, planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

        # 注意力机制选择
        self.use_attention = use_attention
        if use_attention == 'se':
            self.att = ChannelAttention1d(planes, planes // 16)
        else:
            self.att = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用注意力机制
        if self.att is not None:
            out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1d(base.MemoryModule):
    def __init__(self, block, layers, input_channels=1, inplanes=64, num_classes=4,
                 use_attention='se', att_positions='all', method='chtrace_gate',
                 trace_ratio=8, trace_decay=0.5, trace_tile=512,
                 trace_backward='tiled', repeat_times=3, gate_strength=1.0, gate_floor=0.5, reset_at_window=False):
        super().__init__()
        if method not in ('current', 'ottt', 'chtrace', 'chtrace_gate'):
            raise ValueError('Unknown method: ' + method)
        self.trace_config = dict(method=method, trace_ratio=trace_ratio,
                                 trace_decay=trace_decay, trace_tile=trace_tile,
                                 trace_backward=trace_backward)
        self.repeat_times = int(repeat_times)
        self.reset_at_window = bool(reset_at_window)
        self.gate_strength = float(gate_strength)
        self.gate_floor = float(gate_floor)
        if self.repeat_times < 1 or not 0 <= gate_floor <= 1 or gate_strength < 0:
            raise ValueError('Invalid repeat_times/gate settings')
        self.register_memory('online_step', 0)
        self.register_memory('window_gate', None)
        self.gate_sum = 0.0
        self.gate_count = 0
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.sn1 = SpikeWrapper(SpikeNode(tau=2.0, v_threshold=0.1))
        self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
    
        # 残差层配置
        self.layer1 = self._make_layer(block, 64, layers[0], use_attention, att_positions)
        self.layer2 = self._make_layer(block, 128, layers[1], use_attention, att_positions, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], use_attention, att_positions, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], use_attention, att_positions, stride=2)
    
        # A ninth trace site: the pooled spike activity from the stem enters
        # the first residual convolution. Other analog residual/attention
        # paths use their ordinary current-step spatial gradients.
        self.layer1[0].conv1 = make_trace_conv(self.trace_config, inplanes, 64,
            kernel_size=7, stride=1, padding=3, bias=False)
    
        # 注意力增强的池化层
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
    
        # 分类器
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, planes, blocks, use_attention, att_positions, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
    
        # 根据配置确定当前层是否使用注意力
        current_att = use_attention if att_positions == 'all' else (
            use_attention if att_positions == 'last' and blocks == 1 else None)
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_attention=current_att, trace_config=self.trace_config))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # 如果配置为"last"，则只在最后一个块使用注意力
            current_att = use_attention if att_positions == 'all' else (
                use_attention if att_positions == 'last' and i == blocks - 1 else None)
            layers.append(block(self.inplanes, planes, use_attention=current_att, trace_config=self.trace_config))
    
        return nn.Sequential(*layers)

    def detach(self):
        # SpikingJelly 0.0.0.0.14 uses detach_(), which rejects tensor views.
        # Boundary gates can be views, so detach this module's own memories
        # out-of-place. functional.detach_net separately handles child nodes.
        for name, value in self._memories.items():
            if torch.is_tensor(value):
                self._memories[name] = value.detach()

    def forward(self, x):
        if self.reset_at_window and self.online_step > 0 and self.online_step % self.repeat_times == 0:
            from online_trace import TraceMemory
            for module in self.modules():
                if isinstance(module, (neuron.BaseNode, TraceMemory)):
                    module.reset()
        self._prepare_window_gate(x)
        x = self.conv1(x)
        x = self.bn1(x)
    
        x = self.sn1(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        # 双池化增强特征
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
    
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

import math

class SelectiveResNet(ResNet1d):
    def __init__(self,*args,carry_init=.2,carry_max=.5,carry_novelty_beta=1.,
                 carry_scope='layer4',boundary_mode='selective',**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<carry_init<carry_max<=1 or carry_novelty_beta<0:
            raise ValueError('Invalid selective carry initialization/bound')
        if carry_scope not in ['layer4','layer3_layer4'] or boundary_mode not in ['selective','reset_neurons','reset','carry']:
            raise ValueError('Invalid carry scope/mode')
        self.carry_max=carry_max
        self.carry_novelty_beta=carry_novelty_beta
        self.carry_scope=carry_scope
        self.boundary_mode=boundary_mode
        self.carry_stages=['layer4'] if carry_scope=='layer4' else ['layer3','layer4']
        initial=math.log(carry_init/(carry_max-carry_init))
        self.boundary_logit=nn.Parameter(torch.full((2*len(self.carry_stages),),initial))
        self.register_memory('boundary_descriptor',None)
        self.intervention_reset_history=False
        self.reset_boundary_statistics()

    def reset_boundary_statistics(self):
        self.carry_sum=0.
        self.carry_count=0
        self.carry_gradient_norm_sum=0.
        self.carry_gradient_updates=0

    def boundary_report(self):
        return {'mode':self.boundary_mode,'deep_stages':self.carry_stages,
                'learned_base_retention':(self.carry_max*self.boundary_logit.detach().sigmoid()).cpu().tolist(),
                'training_actual_retention_mean':float(self.carry_sum/self.carry_count) if self.carry_count else None,
                'gate_gradient_mean_norm':self.carry_gradient_norm_sum/self.carry_gradient_updates if self.carry_gradient_updates else None,
                'credit_scope':'current boundary loss through detached past membrane; no historical backbone graph'}

    def _descriptor(self,x):
        value=x.detach()
        return torch.cat([F.adaptive_avg_pool1d(value,32),
                          F.adaptive_avg_pool1d(value.square(),32).sqrt()],dim=-1)

    def _prepare_window_gate(self,x):
        # 每窗 repeat_times 步。先处理边界，再做当前步骤的普通 forward。
        # reset_neurons：所有 LIF 清空；深层 q 仍按原 V2 规则保留。
        # reset：LIF 与全部 q 均清空，用于更强的跨窗历史移除对照。
        # q 的门乘法被 detach，故 reset_neurons 对照的门无可学习前向路径。
        # 主方法的门仍通过新窗口第一步的当前损失得到梯度。
        if self.online_step % self.repeat_times==0:
            descriptor=self._descriptor(x)
            difference=x.new_zeros(x.shape[0])
            if self.boundary_descriptor is not None:
                difference=(descriptor-self.boundary_descriptor).square().mean((1,2)).sqrt()
                scale=(descriptor.square().mean((1,2))+self.boundary_descriptor.square().mean((1,2))).sqrt().clamp_min(1e-6)
                difference=difference/scale
            self.boundary_descriptor=descriptor
            if self.online_step>0:
                mode='reset' if self.intervention_reset_history else self.boundary_mode
                if mode!='carry':
                    named=dict(self.named_modules())
                    retained={f'{stage}.{block}.sn1.node':index
                              for index,(stage,block) in enumerate((stage,block) for stage in self.carry_stages for block in range(2))}
                    attenuation=torch.exp(-self.carry_novelty_beta*difference)
                    gate=self.carry_max*self.boundary_logit.sigmoid()
                    for name,module in named.items():
                        if isinstance(module,neuron.BaseNode):
                            if mode=='selective' and name in retained and torch.is_tensor(module.v):
                                alpha=(gate[retained[name]]*attenuation).view(-1,1,1)
                                module.v=alpha*module.v.detach()
                                if self.training:
                                    self.carry_sum=self.carry_sum+alpha.detach().sum()
                                    self.carry_count+=alpha.numel()
                            else:
                                module.reset()
                        elif isinstance(module,TraceMemory):
                            stage=next((stage for stage in self.carry_stages if name.startswith(stage+'.')),None)
                            if mode in ('selective','reset_neurons') and stage is not None and module.q is not None:
                                # TraceConv.prepare deliberately treats history as
                                # detached. Trace carry follows the membrane gate,
                                # but does not add a meta-gradient for gate parameters.
                                block=int(name.split('.')[1]);index=self.carry_stages.index(stage)*2+block
                                alpha=(gate[index]*attenuation).detach().view(-1,1,1)
                                module.q=alpha*module.q.detach()
                            else:
                                module.reset()
        # Do not apply the old global boundary gate a second time.
        # The chosen states/traces were gated above; within-window decay is unchanged.
        if self.training:
            self.window_gate=x.new_ones((x.shape[0],1,1))
            for module in self.modules():
                if isinstance(module,TraceConv1d):module.gate=self.window_gate
        self.online_step+=1


def resnet18(**kwargs):
    return SelectiveResNet(BasicBlock1d,[2,2,2,2],use_attention='se',att_positions='all',**kwargs)


def from_config(c,channels,device):
    if c.get('model_revision') not in ('event_chtrace_v2','event_chtrace_v2_clean','event_chtrace_v3_streamed'):
        raise ValueError('Unsupported model revision for this package')
    keys=['method','trace_ratio','trace_decay','trace_tile','trace_backward','repeat_times',
          'gate_strength','gate_floor','reset_at_window','carry_init','carry_max',
          'carry_novelty_beta','carry_scope','boundary_mode']
    return resnet18(input_channels=channels,num_classes=c['num_classes'],**{k:c[k] for k in keys}).to(device)
