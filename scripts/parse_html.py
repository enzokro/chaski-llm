from bs4 import BeautifulSoup
from fastcore.xtras import Path
from chaski.utils.path_utils import get_project_root

# path to content
base_dir = "/Users/cck/Downloads"
base_dir = Path(base_dir)

# find all relevant Figma course files
lessons = base_dir.glob("Lesson*Help Center.html")

# where to save the content
out_dir = get_project_root() / "data/figma_documents"

for fid in lessons:
    # Read the HTML file
    with open(fid, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the main subject container
    main_subject = soup.find("div", class_="article-body")

    # Extract all paragraph text
    paragraphs = main_subject.find_all("p")
    # for paragraph in paragraphs:
    #     print(paragraph.get_text())
    # join all paragraphs
    content = "\n".join(p.get_text() for p in paragraphs)

    # clean output name for the doc
    name = str(fid.name).replace("– Figma Learn - Help Center.html", "")
    name = name.replace("_", "")
    name = "_".join(name.split())
    with open(out_dir / f"FigmaCourse_{name}.txt", "w") as file:
        file.write(content)
