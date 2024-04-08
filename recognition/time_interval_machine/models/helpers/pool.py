import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, a_dim=128, v_dim=512, hidden_size=512, map_size=49):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(a_dim, hidden_size)
        self.affine_video = nn.Linear(v_dim, hidden_size)
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_h = nn.Linear(map_size, 1, bias=False)

        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        init.xavier_uniform_(self.affine_audio.weight)
        init.constant_(self.affine_audio.bias, 0)
        init.xavier_uniform_(self.affine_video.weight)
        init.constant_(self.affine_video.bias, 0)

    def forward(self, audio, video):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        V_DIM = video.size(-1)
        v_t = video.view(video.size(0) * video.size(1), -1, V_DIM) # [bs*10, 49, 512]
        V = v_t
        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t)) # [bs*10, 49, 512]
        a_t = audio.view(-1, audio.size(-1)) # [bs*10, 128]
        a_t = self.relu(self.affine_audio(a_t)) # [bs*10, 512]

        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2) # [bs*10, 49, 49] + [bs*10, 49, 1]

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2) # [bs*10, 49]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map, [bs*10, 1, 49]
        c_t = torch.bmm(alpha_t, V).view(-1, V_DIM) # [bs*10, 1, 512]
        video_t = c_t.view(video.size(0), -1, V_DIM) # attended visual features, [bs, 10, 512]
        return video_t