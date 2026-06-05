from __future__ import annotations

import html
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".omc" / "tmp_ppt_plan" / "template.pptx"
OUT_DIR = ROOT / "outputs" / "manual-20260602-ppt5" / "presentations" / "defense-first5" / "output"
OUT = OUT_DIR / "traffic-defect-defense-first5.pptx"
COVER_IMG = ROOT / "thesis-images" / "val_batch1_pred.jpg"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
for prefix, uri in NS.items():
    if prefix not in {"ct", "rel"}:
        ET.register_namespace(prefix, uri)


def q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def read_xml(z: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(z.read(name))


def write_xml(z: zipfile.ZipFile, name: str, root: ET.Element) -> None:
    z.writestr(name, ET.tostring(root, encoding="utf-8", xml_declaration=True))


def shape_id(sp: ET.Element) -> str | None:
    cnv = sp.find(".//p:cNvPr", NS)
    return cnv.get("id") if cnv is not None else None


def clear_text_body(txbx: ET.Element) -> None:
    for child in list(txbx):
        if child.tag == q("a", "p"):
            txbx.remove(child)


def make_para(text: str, template_para: ET.Element | None) -> ET.Element:
    para = ET.Element(q("a", "p"))
    if template_para is not None:
        ppr = template_para.find("a:pPr", NS)
        if ppr is not None:
            para.append(clone(ppr))
        first_run = template_para.find("a:r", NS)
        run = clone(first_run) if first_run is not None else ET.Element(q("a", "r"))
    else:
        run = ET.Element(q("a", "r"))
    # Keep run properties, replace only the text payload.
    for child in list(run):
        if child.tag == q("a", "t"):
            run.remove(child)
    t = ET.SubElement(run, q("a", "t"))
    t.text = text
    para.append(run)
    return para


def clone(el: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(el, encoding="utf-8"))


def set_shape_text(slide_root: ET.Element, sid: str, lines: list[str] | str) -> None:
    if isinstance(lines, str):
        lines = [lines]
    for sp in slide_root.findall(".//p:sp", NS):
        if shape_id(sp) != sid:
            continue
        txbx = sp.find(".//p:txBody", NS)
        if txbx is None:
            return
        template_para = txbx.find("a:p", NS)
        clear_text_body(txbx)
        for line in lines:
            txbx.append(make_para(line, template_para))
        return


def replace_first_text(slide_root: ET.Element, old: str, new: str) -> None:
    for t in slide_root.findall(".//a:t", NS):
        if t.text == old:
            t.text = new
            return


def edit_slides(tmp: Path) -> None:
    for idx in range(1, 6):
        slide_path = tmp / "ppt" / "slides" / f"slide{idx}.xml"
        root = ET.parse(slide_path).getroot()

        if idx == 1:
            set_shape_text(root, "3", ["基于深度学习的交通零部件缺陷检测系统", "设计与实现"])
            set_shape_text(root, "10", "指导老师：")
            set_shape_text(root, "15", "答辩人：梁曦霖")
        elif idx == 2:
            set_shape_text(root, "16", "研究背景与意义")
            set_shape_text(root, "41", "系统需求分析")
            set_shape_text(root, "67", "系统概要设计")
            set_shape_text(root, "72", "算法改进与实验")
            set_shape_text(root, "77", "详细设计与实现")
            set_shape_text(root, "82", "系统测试")
            set_shape_text(root, "5", "论文总结")
        elif idx == 3:
            set_shape_text(root, "4", "研究背景与意义")
        elif idx == 4:
            set_shape_text(root, "11", "研究背景")
            set_shape_text(root, "44", "01 工业检测需求")
            set_shape_text(root, "45", ["- 交通零部件表面缺陷影响质量与安全", "- 人工检测效率低、主观性强"])
            set_shape_text(root, "27", "02 深度学习发展")
            set_shape_text(root, "31", ["- YOLO 系列兼具检测精度与响应速度", "- 注意力机制提升小缺陷特征表达"])
            set_shape_text(root, "34", "03 现实应用价值")
            set_shape_text(root, "36", ["- 构建可部署的缺陷检测服务平台", "- 支撑检测记录管理与质量追溯"])
        elif idx == 5:
            set_shape_text(root, "4", "研究意义")
            set_shape_text(root, "26", "交通零部件缺陷检测系统的研究意义")
            set_shape_text(root, "31", "对质量检测")
            set_shape_text(root, "32", ["- 提升缺陷识别效率", "- 降低漏检与误检风险"])
            set_shape_text(root, "37", "对算法研究")
            set_shape_text(root, "38", ["- 验证 CBAM 对 YOLOv8 的改进效果", "- 通过消融实验分析多种优化策略"])
            set_shape_text(root, "33", "对工程应用")
            set_shape_text(root, "34", ["- 实现模型部署、单图/批量检测", "- 提供历史记录、统计分析与模型管理"])

        ET.ElementTree(root).write(slide_path, encoding="utf-8", xml_declaration=True)


def trim_to_five_slides(tmp: Path) -> None:
    pres_path = tmp / "ppt" / "presentation.xml"
    pres = ET.parse(pres_path)
    pres_root = pres.getroot()
    sld_id_lst = pres_root.find("p:sldIdLst", NS)
    keep_rel_ids = set()
    if sld_id_lst is not None:
        for sld in list(sld_id_lst):
            rid = sld.get(q("r", "id"))
            slide_no = len(keep_rel_ids) + 1
            if slide_no <= 5:
                keep_rel_ids.add(rid)
            else:
                sld_id_lst.remove(sld)
    pres.write(pres_path, encoding="utf-8", xml_declaration=True)

    rels_path = tmp / "ppt" / "_rels" / "presentation.xml.rels"
    rels = ET.parse(rels_path)
    rels_root = rels.getroot()
    for rel in list(rels_root):
        target = rel.get("Target", "")
        rid = rel.get("Id")
        if target.startswith("slides/slide") and rid not in keep_rel_ids:
            rels_root.remove(rel)
    rels.write(rels_path, encoding="utf-8", xml_declaration=True)

    content_path = tmp / "[Content_Types].xml"
    content = ET.parse(content_path)
    content_root = content.getroot()
    for override in list(content_root):
        part = override.get("PartName", "")
        m = re.match(r"/ppt/slides/slide(\d+)\.xml", part)
        if m and int(m.group(1)) > 5:
            content_root.remove(override)
    content.write(content_path, encoding="utf-8", xml_declaration=True)

    slides_dir = tmp / "ppt" / "slides"
    rels_dir = slides_dir / "_rels"
    for slide in slides_dir.glob("slide*.xml"):
        m = re.match(r"slide(\d+)\.xml", slide.name)
        if m and int(m.group(1)) > 5:
            slide.unlink()
    for rel in rels_dir.glob("slide*.xml.rels"):
        m = re.match(r"slide(\d+)\.xml\.rels", rel.name)
        if m and int(m.group(1)) > 5:
            rel.unlink()


def replace_cover_image(tmp: Path) -> None:
    if not COVER_IMG.exists():
        return
    media_dir = tmp / "ppt" / "media"
    media_dir.mkdir(exist_ok=True)
    target_name = "codex_cover_defect.jpg"
    shutil.copyfile(COVER_IMG, media_dir / target_name)

    rels_path = tmp / "ppt" / "slides" / "_rels" / "slide1.xml.rels"
    rels = ET.parse(rels_path)
    root = rels.getroot()
    # Picture 1 on the cover uses rId3. Keep the inherited shape, swap media.
    for rel in root:
        if rel.get("Id") == "rId3":
            rel.set("Target", f"../media/{target_name}")
    rels.write(rels_path, encoding="utf-8", xml_declaration=True)

    content_path = tmp / "[Content_Types].xml"
    content = ET.parse(content_path)
    content_root = content.getroot()
    has_jpg = any(el.tag == q("ct", "Default") and el.get("Extension") in {"jpg", "jpeg"} for el in content_root)
    if not has_jpg:
        default = ET.Element(q("ct", "Default"))
        default.set("Extension", "jpg")
        default.set("ContentType", "image/jpeg")
        content_root.insert(0, default)
    content.write(content_path, encoding="utf-8", xml_declaration=True)


def zip_dir(src_dir: Path, out: Path) -> None:
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in src_dir.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(src_dir).as_posix())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = OUT_DIR.parent / "work_unzipped"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(SRC) as z:
        z.extractall(work)
    trim_to_five_slides(work)
    edit_slides(work)
    replace_cover_image(work)
    zip_dir(work, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
