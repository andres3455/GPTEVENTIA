from hf_client import generate_text

prompt = "Escribe una descripción corta, clara y natural sobre un evento relajante en la naturaleza.Debe ser de 2–3 oraciones, tono calmado, sensorial y positivo."

print("=== PROBANDO GENERACIÓN ===")
res = generate_text(prompt, 80)
print(res)
