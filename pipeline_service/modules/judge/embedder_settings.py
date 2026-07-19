from __future__ import annotations

from pydantic import BaseModel


class EmbedderConfig(BaseModel):

    enabled: bool = True
    model_id: str = "Fluxmire/dinov3-vits16-pretrain-lvd1689m"
    revision: str = "main"
    hf_token: str | None = None
    device: str | None = None  
    batch_size: int = 8
    trust_remote_code: bool = False
