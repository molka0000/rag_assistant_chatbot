import os


def prettify_source(source):
    document = os.path.basename(source.get("document"))
    score = source.get("score")
    content_preview = source.get("content_preview")
    #nouveaux champs
    full_length = source.get("full_content_length", "N/A")
    page = source.get("page", "N/A")
    #return f"• **{document}** with score ({round(score,2)}) \n\n **Preview:** \n {content_preview} \n"
    return f"""• **{document}** with score ({round(score,2)}) 
 **Page:** {page} | **Chunk length:** {full_length} characters
 **Preview:** 
 {content_preview} 
"""
