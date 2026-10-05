import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def is_black_rgb(value: str) -> bool:
    return value.strip().lower() in {
        "black",
        "#000",
        "#000000",
        "rgb(0%, 0%, 0%)",
        "rgb(0,0,0)",
        "rgb(0, 0, 0)",
    }


def patch_svg(path):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    root = tree.getroot()

    style_element = ET.fromstring("""
    <style>
      .fill { fill: black; }
      .stroke { stroke: black; }
      @media (prefers-color-scheme: dark) {
        .fill { fill: white; }
        .stroke { stroke: white; }
      }
    </style>
    """)
    root.insert(0, style_element)

    for elem in root.iter():
        classes = []
        fill = elem.attrib.get("fill")
        if fill and is_black_rgb(fill):
            del elem.attrib["fill"]
            classes.append("fill")
        stroke = elem.attrib.get("stroke")
        if stroke and is_black_rgb(stroke):
            del elem.attrib["stroke"]
            classes.append("stroke")
        if classes:
            prev_class = elem.attrib.get("class", "")
            elem.attrib["class"] = f"{prev_class} {' '.join(classes)}".strip()

    tree.write(path)


def main():
    site = Path(__file__).parent
    pdfs = site / "figures"
    svgs = site / "docs" / "fig"
    svgs.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdfs.glob("*.pdf")):
        svg = svgs / f"{pdf.stem}.svg"
        # Check if destination file is older than source file
        if svg.exists() and svg.stat().st_mtime > pdf.stat().st_mtime:
            continue
        subprocess.run(["iperender", "-svg", pdf, svg], check=True)
        print(f"Generated {svg.name}")
        patch_svg(svg)

    # Now, delete svg files that don't have a corresponding pdf or plot
    for svg in sorted(svgs.glob("*.svg")):
        pdf = pdfs / f"{svg.stem}.pdf"
        plot = site / "plots" / f"{svg.stem}.py"
        if not pdf.exists() and not plot.exists():
            svg.unlink()
            print(f"Deleted {svg.name}")


if __name__ == "__main__":
    main()
