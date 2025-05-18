from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
import torch.nn as nn
# class TextEncoder(): # torch.nn.Module
#     def __init__(self, clip_path="openai/clip-vit-base-patch32", device='cuda' , shared_language_projection=False, hidden_dim=512, dropout=0.1):
#         super().__init__()
#         self.device = device
#         self.clip_model = CLIPTextModelWithProjection.from_pretrained(clip_path).to(device)
#         self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path)
#         self.mlp = None
#         if shared_language_projection:
#             self.mlp = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(512, hidden_dim, bias=True),
#                 nn.SiLU(),
#                 nn.Dropout(dropout), # TODO: make this a parameter
#                 nn.Linear(hidden_dim, hidden_dim, bias=True),
#             ).to(self.device)
#             # self.mlp = nn.Sequential(
#             #     nn.Linear(512, hidden_dim, bias=True),
#             #     nn.SiLU(),
#             #     nn.Linear(hidden_dim, hidden_dim, bias=True),
#             # ) # for 17, 18
            
#         self.initialize_weights()

#     def initialize_weights(self):
#         if self.mlp:
#             nn.init.xavier_uniform_(self.mlp[1].weight)
#             nn.init.constant_(self.mlp[1].bias, 0)
#             nn.init.xavier_uniform_(self.mlp[-1].weight)
#             nn.init.constant_(self.mlp[-1].bias, 0)
                
#     def encode(self, text):
#         with torch.no_grad():
#             inputs = self.clip_tokenizer(text, padding=True, return_tensors="pt").to(self.device)
#             outputs = self.clip_model(**inputs)
#             text_embeds = outputs.text_embeds
#         if self.mlp:
#             text_embeds = self.mlp(text_embeds)
#         return text_embeds
    
#     # def parameters(self):
#     #     if self.mlp:
#     #         return self.clip_model.parameters() + self.mlp.parameters()
#     #     else:
#     #         return self.clip_model.parameters()

class ClipTextEncoder(): # torch.nn.Module
    def __init__(self, clip_path="openai/clip-vit-base-patch32", device='cuda'):
        super().__init__()
        self.device = device
        self.clip_model = CLIPTextModelWithProjection.from_pretrained(clip_path).to(device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path)

                
    def encode(self, text):
        with torch.no_grad():
            inputs = self.clip_tokenizer(text, padding=True, return_tensors="pt").to(self.device)
            outputs = self.clip_model(**inputs)
            text_embeds = outputs.text_embeds
        return text_embeds
    
    def parameters(self):
        return self.clip_model.parameters()



class TextEncoder(nn.Module): # torch.nn.Module
    def __init__(self, clip_path="openai/clip-vit-base-patch32", device='cuda' , shared_language_projection=False, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.clip = ClipTextEncoder(clip_path, device)
        self.mlp = None
        if shared_language_projection:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(512, hidden_dim, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout), # TODO: make this a parameter
                nn.Linear(hidden_dim, hidden_dim, bias=True),
            ).to(device)
            # self.mlp = nn.Sequential(
            #     nn.Linear(512, hidden_dim, bias=True),
            #     nn.SiLU(),
            #     nn.Linear(hidden_dim, hidden_dim, bias=True),
            # )
            
        self.initialize_weights()

    def initialize_weights(self):
        if self.mlp:
            nn.init.xavier_uniform_(self.mlp[1].weight)
            nn.init.constant_(self.mlp[1].bias, 0)
            nn.init.xavier_uniform_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, 0)
                
    def encode(self, text):
        with torch.no_grad():
            text_embeds = self.clip.encode(text)
        if self.mlp:
            text_embeds = self.mlp(text_embeds)
        # if self.mlp:
        #     text_embeds_proj = self.mlp(text_embeds)
        #     text_embeds = text_embeds + text_embeds_proj
        return text_embeds
