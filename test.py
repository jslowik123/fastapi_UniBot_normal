import numpy as np
from pinecon_con import PineconeCon

con = PineconeCon("quickstart")

pdf = PyPDF2.PdfReader(file.file)
        text = "".join(page.extract_text() for page in pdf.pages)
        data = [{
            'file': file.filename,
            'content': text
        }]

        embedding = con.create_embeddings(data)
        con.upload_embeddings(data, embedding, namespace= request.namespace)