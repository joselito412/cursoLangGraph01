from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

def filter_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Toma un historial lleno de metadata y estructuras complejas, extrae SOLO el texto
    y devuelve una lista de mensajes nuevos y limpios (sin IDs, sin usage_metadata, etc).
    """
    clean_history = []

    for msg in messages:
        text_content = ""

        # --- PASO 1: EXTRAER SOLO EL TEXTO ---
        if isinstance(msg.content, str):
            # Si ya es texto (ej: "hola"), lo tomamos directo
            text_content = msg.content
        elif isinstance(msg.content, list):
            # Si es una lista compleja (ej: tu JSON con file_search_call),
            # iteramos y sacamos SOLO el valor de la llave "text".
            extracted_texts = [
                block.get("text", "") 
                for block in msg.content 
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            # Unimos los textos (por si el modelo respondió en varios fragmentos)
            text_content = " ".join(extracted_texts)

        # --- PASO 2: CREAR UN NUEVO OBJETO LIMPIO ---
        # No reutilizamos 'msg'. Creamos uno nuevo para asegurar que NO tenga metadata.
        if text_content.strip(): # Solo si hay texto
            if msg.type == "human":
                clean_history.append(HumanMessage(content=text_content))
            elif msg.type == "ai":
                clean_history.append(AIMessage(content=text_content))
            elif msg.type == "system":
                clean_history.append(SystemMessage(content=text_content))
            
            # Nota: Ignoramos intencionalmente los mensajes tipo 'tool' o 'function'
            # ya que pediste "solo los textos de la conversación".

    return clean_history