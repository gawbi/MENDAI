"""검증한 백본 revision에 LoRA와 2출력 분류층을 추가한다."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import AutoModel


class DamageDINO(nn.Module):
    def __init__(self, config, backbone_path=None, checkpointing=False):
        super().__init__()
        source = str(backbone_path or config["model_id"])
        kwargs = {"attn_implementation": "sdpa"}
        if backbone_path:
            if Path(source).name != config["revision"]:
                raise ValueError("캐시 snapshot과 revision 불일치")
            kwargs["local_files_only"] = True
        else:
            kwargs["revision"] = config["revision"]
        backbone = AutoModel.from_pretrained(source, **kwargs)
        if backbone.config.patch_size != 14 or backbone.config.hidden_size != 768:
            raise ValueError("예상한 DINOv2-base 구조가 아님")
        self.targets = [name for name, layer in backbone.named_modules()
                        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj", "query", "value")]
        if len(self.targets) != 24:
            raise ValueError(f"LoRA 적용 대상이 24개가 아님: {len(self.targets)}")
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules=self.targets, bias="none"))
        if checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Linear(2304, 2)

    def forward(self, pixels):
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        patches = hidden[:, 1:]
        features = torch.cat([hidden[:, 0], patches.mean(1), patches.amax(1)], dim=1)
        return self.head(features)

    def trainable_state(self):
        return {"adapter": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.backbone).items()},
                "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()}}

    def load_trainable(self, state):
        if set(state["adapter"]) != set(get_peft_model_state_dict(self.backbone)):
            raise ValueError("LoRA 어댑터 키 불일치")
        result = set_peft_model_state_dict(self.backbone, state["adapter"])
        if result.unexpected_keys:
            raise ValueError(f"어댑터 키 불일치: {result.unexpected_keys}")
        self.head.load_state_dict(state["head"], strict=True)


def optimizer_for(model, config):
    return torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config["lora_lr"]},
        {"params": model.head.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
