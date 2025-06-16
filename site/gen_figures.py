import os
import xml.etree.ElementTree as ET


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
    os.system("mkdir -p docs/fig")
    for filename in os.listdir("figures"):
        if filename.endswith(".pdf"):
            src_path = f"figures/{filename}"
            dst_path = f"docs/fig/{filename}".replace(".pdf", ".svg")
            # Check if destination file is older than source file
            if os.path.exists(dst_path) and os.path.getmtime(
                dst_path
            ) > os.path.getmtime(src_path):
                continue
            os.system(f"iperender -svg {src_path} {dst_path}")
            print(f"Generated {dst_path}")
            patch_svg(dst_path)

    # Now, delete svg files that don't have a corresponding pdf
    for filename in os.listdir("docs/fig"):
        if filename.endswith(".svg"):
            src_path = f"docs/fig/{filename}"
            dst_path = f"figures/{filename}".replace(".svg", ".pdf")
            if not os.path.exists(dst_path):
                os.remove(src_path)
                print(f"Deleted {src_path}")


if __name__ == "__main__":
    main()
