from pydantic import BaseModel


class RendererConfig(BaseModel):
    sidecar_host: str = "127.0.0.1"
    sidecar_port: int = 8003
    startup_timeout_s: float = 60.0
    request_timeout_s: float = 45.0
    node_binary: str = "node"
    img_size: int = 518
    grid_gap: int = 5
    bg_color: str | None = None
    sidecar_count: int = 4
    pool_size: int = 2
    judge_multiview: bool = True
    judge_white_bg: str = "ffffff"
    judge_gray_bg: str = "808080"
    render_timeout_ms: int = 60000
    protocol_timeout_ms: int = 60000
    static_port_base: int = 3000
