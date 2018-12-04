import torch.nn.functional as F
import torch
import numpy as np
a = torch.Tensor(10, 2, 2, 3)
b = torch.Tensor(10, 10, 3, 2)
print(torch.matmul(a, b).shape)


'''

attn_matrix = torch.empty(Q.size(0), Q.size(1), Q.size(1), Q.size(-2), K.size(-2))
            attn_matrix = attn_matrix.cuda()
            for h in range(num_heads):
                for s in range(num_heads):
                    attn_matrix[:, h, s] = torch.matmul(Q[:, h], K[:, s].transpose(-2, -1)) / math.sqrt(Q.size(-1))
            for h in range(num_heads):
                head_low = max(0, h - self.head_width)
                head_high = min(num_heads, h + self.head_width)
                add = 0
                for k in range(head_low, head_high):
                    add = attn_matrix[:, h, k] + add
                if mask is not None:
                    add = add.masked_fill(mask == 0, -1e30)
                if element_mask is not None:
                    add = add.masked_fill(element_mask == 0, -1e30)
                add = torch.sum(torch.exp(add) + 1e-10, -1).unsqueeze(-1)
                for s in range(num_heads):
                    if mask is not None:
                        matrix = attn_matrix[:, h, s].masked_fill(mask == 0, -1e30)
                    else:
                        matrix = attn_matrix[:, h, s]
                    if element_mask is not None:
                        matrix = matrix.masked_fill(element_mask == 0, -1e30)
                    attn_matrix[:, h, s] = (torch.exp(matrix) + 1e-10) / add  # [nb, len1, len2]
                    if element_mask is not None:
                        attn_matrix[:, h, s] = attn_matrix[:, h, s].masked_fill(element_mask == 0, 0)
            # V: [nb, nh, len2, hid2]
            out = torch.empty(V.size(0), num_heads, Q.size(-2), V.size(-1))
            out = out.cuda()
            for h in range(num_heads):
                head_low = max(0, h - self.head_width)
                head_high = min(num_heads, h + self.head_width)
                add = 0
                for s in range(head_low, head_high):
                    add = torch.matmul(attn_matrix[:, h, s], V[:, s]) + add
                out[:, h] = add
'''