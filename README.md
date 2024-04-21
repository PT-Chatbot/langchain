![Banner](./LangChain%20Banner.webp)

# Langchain

Langchain es una poderosa biblioteca diseñada para transformar la interacción y el flujo de trabajo con modelos de lenguaje de aprendizaje automático. A continuación, se ofrece una visión general de los temas y cuadernos incluidos en este repositorio, destacando cómo cada uno aborda diferentes aspectos y funcionalidades de Langchain.

## Cuadernos y temas clave

### Introducción a LangChain

#### **1. Preparación**

Esta sección se encarga de la instalación de `langchain` y `openai`, seguida de la configuración de la clave API de OpenAI en el entorno.

```python
%pip install langchain
%pip install openai
```

```python
import config
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
```

#### **2. Introducción a LangChain**

Se exponen los conceptos básicos de LangChain, incluyendo Model, Prompt Templates y Chain, aunque en esta sección no se incluye código específico.

#### **3. Modelos**

##### **3.1 LLMs (Language Learning Models)**

Muestra cómo utilizar los modelos de OpenAI en LangChain, desde la inicialización hasta la generación de texto y la ejecución de ejemplos específicos.

```python
from langchain.llms import OpenAI
llm = OpenAI()
# Ejemplo de uso: llm.predict('Cuéntame un chiste.')
```

##### **3.2 ChatModels**

Explica cómo los ChatModels procesan una lista de mensajes, detallando los roles de cada entidad en la comunicación. El código muestra la creación de mensajes y la generación de plantillas para la interacción.

```python
from langchain.schema.messages import ChatMessage
# Ejemplo de creación de chat history y generación de una respuesta.
```

#### **4. Chain**

Demuestra cómo crear y utilizar una cadena en LangChain, incluyendo la creación de una plantilla de prompt y la ejecución de una cadena para generar nombres de empresas basados en una descripción.

```python
from langchain import PromptTemplate
from langchain.chains import LLMChain
# Ejemplo de creación y ejecución de una cadena.
```

### Langchain 1 - Modelos y Prompts

#### **1. Modelos**

Esta sección demuestra cómo cargar e interactuar con modelos de lenguaje de OpenAI utilizando LangChain, enfocándose en cómo se puede solicitar a estos modelos que generen texto.

```python
from langchain.llms import LlamaCpp, OpenAI
llm_openai = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=api)
respuesta_openai = llm_openai("Hola, como estas?")
print(respuesta_openai)
```

#### **2. Modelos Chat**

##### **2.1 ChatGTP**

Se introduce el uso de modelos específicos para chat, como ChatGPT, explicando cómo se pueden usar para generar respuestas más contextualizadas en diálogos.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
chatgpt = ChatOpenAI(openai_api_key=api)
respuesta = chatgpt([HumanMessage(content="Hola, como estas?")])
print(respuesta.content)
```

#### **3. Prompts**

Explica la importancia de estructurar bien los prompts y cómo LangChain ofrece herramientas para facilitar esto, permitiendo la creación de templates y la incorporación de ejemplos para mejorar la precisión de las respuestas.

```python
from langchain import PromptTemplate
template_basico = """Eres un asistente virtual culinario...
prompt_temp = PromptTemplate(input_variables=["platillo"], template = template_basico)
```

##### **3.1 Ejemplo cuando no usamos PROMP de ejemplo**

Muestra la diferencia en la respuesta del modelo al no utilizar un prompt de ejemplo, destacando la utilidad de estos en proporcionar contexto al modelo.

```python
llm_openai("¿Cuál es el ingrediente principal de las quesadillas?")
```

#### 4. Output parser

LangChain permite parsear o formatear las respuestas del modelo para que sean más útiles o adecuadas para el contexto deseado.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
template_basico_parser = "Cuales son los ingredientes para preparar {platillo}\n{como_parsear}"
```

### Langchain 2 - Memoria

#### **1. Memoria**

Aqui se introduce el concepto de memoria en los LLMs, explicando cómo ayuda a los modelos a recordar información a largo plazo y describiendo diferentes tipos de memoria que se pueden implementar.

#### **2. Conversation buffer**

Esta memoria básica almacena cada interacción con el modelo. El historial completo se envía con cada nuevo prompt para ayudar al modelo a recordar interacciones previas.

```python
from langchain.memory import ConversationBufferMemory
memoria = ConversationBufferMemory()
chatbot = ConversationChain(llm=llm, memory=memoria, verbose=True)
chatbot.predict(input="Hola como estas?, Me llamo Marcos y soy un programador")
```

#### **3. Conversation buffer window memory**

Similar a la memoria del buffer de conversación, pero limita el historial a una ventana específica de mensajes, evitando sobrecargar el modelo con información innecesaria.

```python
from langchain.memory import ConversationBufferWindowMemory
memoria = ConversationBufferWindowMemory(window_size=5)  # Ejemplo de tamaño de ventana
```

#### **4. Conversation summary memory**

En lugar de enviar todo el historial, esta memoria envía un resumen de la conversación, permitiendo que el modelo recuerde el contexto sin sobrepasar su límite de tokens.

```python
from langchain.memory import ConversationSummaryMemory
memoria = ConversationSummaryMemory(llm=llm)
chatbot_resumen = ConversationChain(llm=llm, memory=memoria, verbose=True)
```

##### **4.1 Resumen de la conversación**

Demuestra cómo la memoria de resumen de conversación solo envía un resumen en lugar del historial completo al modelo.

```python
memoria.chat_memory.messages  # Muestra los mensajes en memoria
```

#### **5. Conversation Knowledge Graph Memory**

Implementa un grafo de conocimiento, almacenando piezas clave de la conversación para que el modelo pueda referenciar y responder con base en ese contexto.

```python
from langchain.memory import ConversationKGMemory
memoria = ConversationKGMemory(llm=llm)
chatbot_kgm = ConversationChain(llm=llm, memory=memoria, verbose=True)
```

##### **5.1 Detalles del grafo de conocimiento**

Muestra cómo se almacena la información clave en un grafo de conocimiento.

```python
print(chatbot_kgm.memory.kg.get_triples())  # Muestra los triples almacenados en el grafo
```

### Langchain 3 - Cadenas

#### **1. Cadenas**

Esta sección introduce el concepto de cadenas en LangChain, que permite crear flujos de trabajo combinando distintos "bloques" para crear sistemas con LLMs más complejos. Se menciona cómo las cadenas permiten gestionar qué modelo genera qué información, cómo se utilizan los prompts y cómo la salida de un modelo puede funcionar como entrada para otro.

#### **2. Cadenas más usadas**

Se presentan las cadenas más comunes y útiles integradas en LangChain, que facilitan el desarrollo de diversos sistemas.

##### **2.1 LLMChain**

Describe cómo LLMChain facilita la interacción con LLMs combinando un modelo y los templates de prompts. Se muestra cómo se puede utilizar LLMChain para generar respuestas en base a un tema específico.

```python
from langchain import LLMChain, OpenAI, PromptTemplate
llm = OpenAI(openai_api_key=API)
cadena_LLM = LLMChain(llm=llm, prompt=template)
cadena_LLM.predict(tema="ingenieria civil")
```

##### **2.2 SequentialChain**

Explica cómo SequentialChain permite crear secuencias de operaciones donde la salida de una cadena se convierte en la entrada de la siguiente, proporcionando un ejemplo de cómo se pueden encadenar dos procesos para obtener una recomendación de aprendizaje.

```python
from langchain.chains import SequentialChain
cadenas = SequentialChain(chains=[cadena_lista, cadena_inicio])
cadenas({"tema": "programacion"})
```

#### **3. Otros ejemplos**

Se proporcionan ejemplos adicionales de tipos de cadenas en LangChain, como MathChain y TransformChain, mostrando cómo se pueden realizar operaciones matemáticas o transformaciones de texto.

```python
# Ejemplo de MathChain
from langchain import LLMMathChain
cadena_mate = LLMMathChain(llm=llm, verbose=True)
cadena_mate.run("Cuanto es 432*12-32+32?")

# Ejemplo de TransformChain
from langchain.chains import TransformChain
cadena_transformacion = TransformChain(transform=eliminar_brincos)
cadena_transformacion.run(prompt)
```
