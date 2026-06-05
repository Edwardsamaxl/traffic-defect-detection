from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "system-overall-architecture.png"
SVG_PATH = OUT_DIR / "system-overall-architecture.svg"

W, H = 1800, 1120


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(42, True)
FONT_LAYER = font(29, True)
FONT_H2 = font(26, True)
FONT_H3 = font(22, True)
FONT_BODY = font(18)
FONT_SMALL = font(15)


COLORS = {
    "bg": "#f7f8fb",
    "ink": "#1e293b",
    "muted": "#5b6472",
    "panel_border": "#c9d3e2",
    "frontend": "#e8f6f3",
    "frontend_dark": "#127c72",
    "service": "#eef3ff",
    "service_dark": "#3158a8",
    "data": "#fff3e6",
    "data_dark": "#b85a12",
    "box": "#ffffff",
    "line": "#637083",
    "accent": "#d74b3f",
    "green": "#2d7d46",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return (*hex_to_rgb(value), alpha)


def draw_round(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str,
    width: int = 2,
    radius: int = 18,
) -> None:
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = COLORS["ink"],
    line_gap: int = 5,
) -> None:
    x1, y1, x2, y2 = xy
    max_width = x2 - x1 - 28
    lines: list[str] = []
    for raw in text.split("\n"):
        if d.textlength(raw, font=fnt) <= max_width:
            lines.append(raw)
            continue
        line = ""
        for ch in raw:
            if d.textlength(line + ch, font=fnt) <= max_width:
                line += ch
            else:
                if line:
                    lines.append(line)
                line = ch
        if line:
            lines.append(line)
    heights = [d.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2 - 1
    for line, h in zip(lines, heights):
        tw = d.textlength(line, font=fnt)
        d.text((x1 + (x2 - x1 - tw) / 2, y), line, font=fnt, fill=fill)
        y += h + line_gap


def arrow(
    d: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = COLORS["line"],
    width: int = 3,
    label: str | None = None,
    label_offset: tuple[int, int] = (0, 0),
) -> None:
    d.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        pts = [(ex, ey), (ex - 16 * direction, ey - 8), (ex - 16 * direction, ey + 8)]
    else:
        direction = 1 if ey >= sy else -1
        pts = [(ex, ey), (ex - 8, ey - 16 * direction), (ex + 8, ey - 16 * direction)]
    d.polygon(pts, fill=color)
    if label:
        lx = (sx + ex) // 2 + label_offset[0]
        ly = (sy + ey) // 2 + label_offset[1]
        bbox = d.textbbox((0, 0), label, font=FONT_SMALL)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.rounded_rectangle((lx - tw / 2 - 9, ly - th / 2 - 7, lx + tw / 2 + 9, ly + th / 2 + 7), radius=10, fill="#ffffff", outline="#d9e0ea")
        d.text((lx - tw / 2, ly - th / 2 - 1), label, font=FONT_SMALL, fill=color)


def pill(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str,
    outline: str,
    fnt: ImageFont.ImageFont = FONT_BODY,
) -> None:
    draw_round(d, xy, fill, outline, width=2, radius=15)
    centered_text(d, xy, text, fnt)


def draw_png() -> None:
    img = Image.new("RGB", (W, H), hex_to_rgb(COLORS["bg"]))
    d = ImageDraw.Draw(img)

    d.text((W / 2, 38), "钢材表面缺陷检测系统整体架构图", font=FONT_TITLE, fill=COLORS["ink"], anchor="mt")
    d.text(
        (W / 2, 96),
        "以三层职责划分为主线，突出界面交互、业务编排、模型检测与数据持久化之间的协作关系",
        font=FONT_BODY,
        fill=COLORS["muted"],
        anchor="mt",
    )

    panels = [
        ("前端交互层", (70, 145, 1730, 345), COLORS["frontend"], COLORS["frontend_dark"], "提供操作入口与结果呈现"),
        ("后端服务层", (70, 390, 1730, 805), COLORS["service"], COLORS["service_dark"], "承载核心业务逻辑与检测流程"),
        ("数据持久层", (70, 850, 1730, 1040), COLORS["data"], COLORS["data_dark"], "统一保存结构化数据与文件资源"),
    ]

    for title, xy, fill, dark, desc in panels:
        draw_round(d, xy, fill, dark, width=3, radius=28)
        x1, y1, x2, y2 = xy
        d.rounded_rectangle((x1, y1, x1 + 205, y2), radius=28, fill=rgba(dark, 255), outline=dark)
        d.rectangle((x1 + 178, y1, x1 + 205, y2), fill=dark)
        centered_text(d, (x1 + 8, y1 + 20, x1 + 198, y1 + 95), title, FONT_LAYER, "#ffffff")
        centered_text(d, (x1 + 18, y1 + 105, x1 + 188, y2 - 20), desc, FONT_SMALL, "#ffffff")

    # Frontend layer
    pill(d, (305, 185, 545, 245), "检测任务界面", "#ffffff", COLORS["frontend_dark"], FONT_H3)
    pill(d, (590, 185, 830, 245), "历史记录检索", "#ffffff", COLORS["frontend_dark"], FONT_H3)
    pill(d, (875, 185, 1115, 245), "统计信息可视化", "#ffffff", COLORS["frontend_dark"], FONT_H3)
    pill(d, (1160, 185, 1400, 245), "模型管理入口", "#ffffff", COLORS["frontend_dark"], FONT_H3)
    pill(d, (1445, 185, 1625, 245), "登录 / 注册", "#ffffff", COLORS["frontend_dark"], FONT_H3)
    draw_round(d, (360, 275, 1580, 325), "#dff0ec", "#71aaa2", width=2, radius=14)
    centered_text(d, (360, 275, 1580, 325), "统一前端应用：封装页面状态、表单输入、文件上传、结果展示与图表渲染", FONT_BODY, COLORS["ink"])

    # Backend layer
    draw_round(d, (300, 425, 1620, 495), "#ffffff", "#8ea3d6", width=2, radius=18)
    centered_text(d, (300, 425, 1620, 495), "请求接入与权限校验：统一接收前端请求，完成身份校验、参数校验与响应封装", FONT_H3, COLORS["service_dark"])

    module_boxes = [
        ((310, 535, 560, 645), "认证管理", "注册、登录、令牌签发\n用户身份校验"),
        ((600, 535, 850, 645), "智能分析", "单样本检测、批量检测\n结果后处理"),
        ((890, 535, 1140, 645), "数据处理", "历史记录查询\n统计聚合与指标计算"),
        ((1180, 535, 1430, 645), "模型生命周期管理", "权重上传、版本维护\n模型实例复用"),
    ]
    for xy, title, body in module_boxes:
        draw_round(d, xy, "#ffffff", "#566fb2", width=2, radius=18)
        centered_text(d, (xy[0], xy[1] + 10, xy[2], xy[1] + 46), title, FONT_H3, COLORS["service_dark"])
        centered_text(d, (xy[0] + 14, xy[1] + 47, xy[2] - 14, xy[3] - 10), body, FONT_SMALL, COLORS["muted"])

    draw_round(d, (1480, 535, 1620, 645), "#f8fbff", "#566fb2", width=2, radius=18)
    centered_text(d, (1480, 535, 1620, 645), "接口规范\nJSON 响应\n错误处理", FONT_SMALL, COLORS["service_dark"])

    draw_round(d, (420, 690, 1500, 775), "#f9fbff", "#8ea3d6", width=2, radius=18)
    centered_text(d, (420, 696, 1500, 730), "检测任务编排", FONT_H3, COLORS["service_dark"])
    step_w = 190
    steps = ["图像与阈值输入", "模型实例缓存", "单图 / 批量推理", "结果后处理", "结果持久化"]
    for i, step in enumerate(steps):
        x = 470 + i * 198
        pill(d, (x, 732, x + step_w, 762), step, "#ffffff", "#9fb0d8", FONT_SMALL)
        if i < len(steps) - 1:
            arrow(d, (x + step_w + 6, 747), (x + 190 + 198 - 12, 747), "#8a97b0", 2)

    # Data layer
    draw_round(d, (315, 885, 795, 1015), "#ffffff", "#c87525", width=2, radius=22)
    centered_text(d, (315, 895, 795, 932), "关系型数据区", FONT_H3, COLORS["data_dark"])
    for xy, text in [
        ((350, 948, 475, 992), "用户信息"),
        ((495, 948, 635, 992), "检测记录"),
        ((655, 948, 760, 992), "模型元数据"),
    ]:
        pill(d, xy, text, "#fff8f0", "#d99557", FONT_SMALL)

    draw_round(d, (875, 885, 1355, 1015), "#ffffff", "#c87525", width=2, radius=22)
    centered_text(d, (875, 895, 1355, 932), "文件资源区", FONT_H3, COLORS["data_dark"])
    for xy, text in [
        ((905, 948, 1038, 992), "上传图像"),
        ((1060, 948, 1200, 992), "标注结果"),
        ((1220, 948, 1325, 992), "模型权重"),
    ]:
        pill(d, xy, text, "#fff8f0", "#d99557", FONT_SMALL)

    draw_round(d, (1415, 885, 1620, 1015), "#fffaf4", "#c87525", width=2, radius=22)
    centered_text(d, (1415, 895, 1620, 1015), "会话级连接池\n事务管理\n路径索引", FONT_BODY, COLORS["data_dark"])

    # Main flow arrows
    arrow(d, (960, 345), (960, 425), COLORS["accent"], 4, "HTTP 请求 / JSON 响应", (0, -8))
    d.line([(960, 495), (960, 515), (870, 515), (870, 690)], fill=COLORS["accent"], width=4)
    d.polygon([(870, 690), (862, 674), (878, 674)], fill=COLORS["accent"])
    label = "业务调度"
    bbox = d.textbbox((0, 0), label, font=FONT_SMALL)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.rounded_rectangle((814 - tw / 2 - 9, 582 - th / 2 - 7, 814 + tw / 2 + 9, 582 + th / 2 + 7), radius=10, fill="#ffffff", outline="#d9e0ea")
    d.text((814 - tw / 2, 582 - th / 2 - 1), label, font=FONT_SMALL, fill=COLORS["accent"])
    arrow(d, (715, 775), (555, 885), COLORS["green"], 4, "结构化记录", (-40, -8))
    arrow(d, (1115, 775), (1115, 885), COLORS["green"], 4, "文件与检测产物", (80, -8))
    arrow(d, (1310, 775), (1515, 885), COLORS["green"], 4, "模型与事务状态", (45, -8))

    # Cross-layer notes
    d.rounded_rectangle((115, 1060, 1685, 1093), radius=14, fill="#ffffff", outline="#d5dde8", width=1)
    centered_text(
        d,
        (115, 1060, 1685, 1093),
        "设计重点：前端不直接访问模型或数据库；后端统一协调认证、检测、统计与模型管理；持久层同时承载业务数据和检测文件资源。",
        FONT_SMALL,
        COLORS["muted"],
    )

    img.save(PNG_PATH)


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_text(x: int, y: int, text: str, size: int, weight: int = 400, color: str = COLORS["ink"], anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{esc(text)}</text>'


def svg_rect(x: int, y: int, w: int, h: int, fill: str, stroke: str, rx: int = 18, sw: int = 2) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def svg_multiline_center(x: int, y: int, w: int, h: int, text: str, size: int, color: str, weight: int = 400, max_chars: int = 18) -> str:
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(wrap(part, max_chars) or [""])
    start = y + h / 2 - (len(lines) - 1) * size * 0.7
    out = []
    for i, line in enumerate(lines):
        out.append(svg_text(x + w // 2, int(start + i * size * 1.35), line, size, weight, color))
    return "\n".join(out)


def draw_svg() -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        "<style>text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif;dominant-baseline:middle}.small{font-size:15px}</style>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#637083"/></marker>',
        '<marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#d74b3f"/></marker>',
        '<marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2d7d46"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
        svg_text(900, 60, "钢材表面缺陷检测系统整体架构图", 42, 700),
        svg_text(900, 105, "以三层职责划分为主线，突出界面交互、业务编排、模型检测与数据持久化之间的协作关系", 18, 400, COLORS["muted"]),
    ]

    panels = [
        ("前端交互层", 70, 145, 1660, 200, COLORS["frontend"], COLORS["frontend_dark"], "提供操作入口与结果呈现"),
        ("后端服务层", 70, 390, 1660, 415, COLORS["service"], COLORS["service_dark"], "承载核心业务逻辑与检测流程"),
        ("数据持久层", 70, 850, 1660, 190, COLORS["data"], COLORS["data_dark"], "统一保存结构化数据与文件资源"),
    ]
    for title, x, y, w, h, fill, dark, desc in panels:
        parts.append(svg_rect(x, y, w, h, fill, dark, 28, 3))
        parts.append(svg_rect(x, y, 205, h, dark, dark, 28, 0))
        parts.append(f'<rect x="{x+178}" y="{y}" width="27" height="{h}" fill="{dark}"/>')
        parts.append(svg_multiline_center(x + 8, y + 20, 190, 75, title, 29, "#ffffff", 700, 6))
        parts.append(svg_multiline_center(x + 20, y + 115, 165, h - 135, desc, 15, "#ffffff", 400, 8))

    def add_pill(x: int, y: int, w: int, h: int, text: str, stroke: str, fill: str = "#ffffff", size: int = 18, weight: int = 400) -> None:
        parts.append(svg_rect(x, y, w, h, fill, stroke, 15, 2))
        parts.append(svg_multiline_center(x, y, w, h, text, size, COLORS["ink"], weight, 12))

    for x, text, width in [(305, "检测任务界面", 240), (590, "历史记录检索", 240), (875, "统计信息可视化", 240), (1160, "模型管理入口", 240), (1445, "登录 / 注册", 180)]:
        add_pill(x, 185, width, 60, text, COLORS["frontend_dark"], "#ffffff", 22, 700)
    parts.append(svg_rect(360, 275, 1220, 50, "#dff0ec", "#71aaa2", 14, 2))
    parts.append(svg_text(970, 300, "统一前端应用：封装页面状态、表单输入、文件上传、结果展示与图表渲染", 18))

    parts.append(svg_rect(300, 425, 1320, 70, "#ffffff", "#8ea3d6", 18, 2))
    parts.append(svg_text(960, 460, "请求接入与权限校验：统一接收前端请求，完成身份校验、参数校验与响应封装", 22, 700, COLORS["service_dark"]))
    modules = [
        (310, 535, "认证管理", "注册、登录、令牌签发\n用户身份校验"),
        (600, 535, "智能分析", "单样本检测、批量检测\n结果后处理"),
        (890, 535, "数据处理", "历史记录查询\n统计聚合与指标计算"),
        (1180, 535, "模型生命周期管理", "权重上传、版本维护\n模型实例复用"),
    ]
    for x, y, title, body in modules:
        parts.append(svg_rect(x, y, 250, 110, "#ffffff", "#566fb2", 18, 2))
        parts.append(svg_text(x + 125, y + 28, title, 22, 700, COLORS["service_dark"]))
        parts.append(svg_multiline_center(x + 15, y + 50, 220, 50, body, 15, COLORS["muted"], 400, 13))
    parts.append(svg_rect(1480, 535, 140, 110, "#f8fbff", "#566fb2", 18, 2))
    parts.append(svg_multiline_center(1480, 535, 140, 110, "接口规范\nJSON 响应\n错误处理", 15, COLORS["service_dark"], 700, 7))
    parts.append(svg_rect(420, 690, 1080, 85, "#f9fbff", "#8ea3d6", 18, 2))
    parts.append(svg_text(960, 712, "检测任务编排", 22, 700, COLORS["service_dark"]))
    for i, step in enumerate(["图像与阈值输入", "模型实例缓存", "单图 / 批量推理", "结果后处理", "结果持久化"]):
        x = 470 + i * 198
        add_pill(x, 732, 190, 30, step, "#9fb0d8", "#ffffff", 15, 400)
        if i < 4:
            parts.append(f'<line x1="{x+196}" y1="747" x2="{x+186+198}" y2="747" stroke="#8a97b0" stroke-width="2" marker-end="url(#arrow)"/>')

    parts.append(svg_rect(315, 885, 480, 130, "#ffffff", "#c87525", 22, 2))
    parts.append(svg_text(555, 914, "关系型数据区", 22, 700, COLORS["data_dark"]))
    for x, text, width in [(350, "用户信息", 125), (495, "检测记录", 140), (655, "模型元数据", 105)]:
        add_pill(x, 948, width, 44, text, "#d99557", "#fff8f0", 15)
    parts.append(svg_rect(875, 885, 480, 130, "#ffffff", "#c87525", 22, 2))
    parts.append(svg_text(1115, 914, "文件资源区", 22, 700, COLORS["data_dark"]))
    for x, text, width in [(905, "上传图像", 133), (1060, "标注结果", 140), (1220, "模型权重", 105)]:
        add_pill(x, 948, width, 44, text, "#d99557", "#fff8f0", 15)
    parts.append(svg_rect(1415, 885, 205, 130, "#fffaf4", "#c87525", 22, 2))
    parts.append(svg_multiline_center(1415, 885, 205, 130, "会话级连接池\n事务管理\n路径索引", 18, COLORS["data_dark"], 400, 7))

    parts.append('<line x1="960" y1="345" x2="960" y2="425" stroke="#d74b3f" stroke-width="4" marker-end="url(#arrow-red)"/>')
    parts.append(svg_text(960, 382, "HTTP 请求 / JSON 响应", 15, 400, COLORS["accent"]))
    parts.append('<polyline points="960,495 960,515 870,515 870,690" fill="none" stroke="#d74b3f" stroke-width="4" marker-end="url(#arrow-red)"/>')
    parts.append(svg_rect(779, 568, 70, 28, "#ffffff", "#d9e0ea", 10, 1))
    parts.append(svg_text(814, 582, "业务调度", 15, 400, COLORS["accent"]))
    parts.append('<line x1="715" y1="775" x2="555" y2="885" stroke="#2d7d46" stroke-width="4" marker-end="url(#arrow-green)"/>')
    parts.append(svg_text(595, 830, "结构化记录", 15, 400, COLORS["green"]))
    parts.append('<line x1="1115" y1="775" x2="1115" y2="885" stroke="#2d7d46" stroke-width="4" marker-end="url(#arrow-green)"/>')
    parts.append(svg_text(1195, 830, "文件与检测产物", 15, 400, COLORS["green"]))
    parts.append('<line x1="1310" y1="775" x2="1515" y2="885" stroke="#2d7d46" stroke-width="4" marker-end="url(#arrow-green)"/>')
    parts.append(svg_text(1510, 830, "模型与事务状态", 15, 400, COLORS["green"]))
    parts.append(svg_rect(115, 1060, 1570, 33, "#ffffff", "#d5dde8", 14, 1))
    parts.append(svg_text(900, 1077, "设计重点：前端不直接访问模型或数据库；后端统一协调认证、检测、统计与模型管理；持久层同时承载业务数据和检测文件资源。", 15, 400, COLORS["muted"]))

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    draw_png()
    draw_svg()
    print(PNG_PATH)
    print(SVG_PATH)
