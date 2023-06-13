import os


def on_post_build(config):
    print("Tidying HTML files...")
    tidy_cmd = [
        "tidy",
        "-modify",
        "-indent",
        "-quiet",
        "--wrap 0",
        "--drop-empty-elements no",
        "--custom-tags yes",
        "--tidy-mark no",
        "2> /dev/null",
    ]
    for root, _, files in os.walk(config["site_dir"]):
        for file in files:
            if file == "index.html":
                full_path = os.path.join(root, file)
                os.system(" ".join(tidy_cmd + [full_path]))
