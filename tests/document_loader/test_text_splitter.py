from document_loader.format import Format
from document_loader.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import PyPDF2

def load_pdf_page_text(pdf_path: Path, page_number: int) -> str:
    """
    Charge le texte d'une page spécifique d'un PDF avec PyPDF2.
    """
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        if page_number >= len(reader.pages):
            raise ValueError(f"Le PDF n'a que {len(reader.pages)} pages, page demandée: {page_number}")
        page = reader.pages[page_number]
        return page.extract_text() or ""

def test_pdf_splitter_page2():
    """
    Teste le text splitter sur la 2ème page du PDF 'Code monétaire et financier'.
    Affiche les chunks pour visualiser leur contenu.
    """
    # Chemin vers ton PDF (ajuster si nécessaire)
    pdf_path = Path(__file__).parent.parent.parent / "docs" / "reglementation" / "code monétaire et finanacier.pdf"

    # Charger le texte de la 2ème page (index 1)
    text = load_pdf_page_text(pdf_path, page_number=1)

    # Créer le splitter avec les separators de Format.PDF
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,         # taille max d'un chunk
        chunk_overlap=100,       # chevauchement entre chunks
        separators=Format.PDF.value,
        keep_separator=True
    )

    # Découper le texte en chunks
    chunks = splitter.split_text(text)

    # Afficher les chunks pour visualisation
    print("\n=== CHUNKS DE LA 2ÈME PAGE ===\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i} ({len(chunk)} caractères):")
        print(repr(chunk))
        print("---")

    # Vérification simple
    assert len(chunks) > 0, "Aucun chunk n'a été généré."

# Si on veut lancer le test directement
if __name__ == "__main__":
    test_pdf_splitter_page2()
