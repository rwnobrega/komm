import os


def main():
    os.system("mkdir -p docs/figures")
    for filename in os.listdir("figures"):
        if filename.endswith(".pdf"):
            src_path = f"figures/{filename}"
            dst_path = f"docs/figures/{filename}".replace(".pdf", ".svg")
            # Check if destination file is older than source file
            if os.path.exists(dst_path) and os.path.getmtime(
                dst_path
            ) > os.path.getmtime(src_path):
                continue
            os.system(f"iperender -svg {src_path} {dst_path}")
            print(f"Generated {dst_path}")
    # Now, delete svg files that don't have a corresponding pdf
    for filename in os.listdir("docs/figures"):
        if filename.endswith(".svg"):
            src_path = f"docs/figures/{filename}"
            dst_path = f"figures/{filename}".replace(".svg", ".pdf")
            if not os.path.exists(dst_path):
                os.remove(src_path)
                print(f"Deleted {src_path}")


if __name__ == "__main__":
    main()
