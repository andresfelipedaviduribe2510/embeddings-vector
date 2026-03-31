# Explorando la Semántica de la IA: Embeddings y Similitud de Coseno

Este proyecto tiene como objetivo desmitificar cómo las máquinas "entienden" el lenguaje humano. Utilizando el modelo de embeddings de Google Gemini, transformamos palabras en vectores matemáticos para comparar su significado mediante cálculos de álgebra lineal.

## Tecnologías Utilizadas

- **Node.js**: Para la obtención de embeddings mediante la SDK de `@google/generative-ai`.
- **Python (Google Colab)**: Para el análisis numérico y cálculo de similitudes.
- **NumPy**: Biblioteca matemática para manejar vectores y realizar operaciones de álgebra lineal.

## Paso 1: Generación de Embeddings (Node.js)

En esta etapa, enviamos una palabra a la API de Gemini y recibimos un vector (una lista de números decimales). Este vector representa la posición de la palabra en un espacio multidimensional de conceptos.
````js
import "dotenv/config";
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

const { embedding } = await model.embedContent("Casa");

console.log(embedding.values);
// Imprime una lista de números que representan el "ADN semántico" de la palabra.
` `` `

## Paso 2: Análisis de Similitud (Python / Colab)

Una vez obtenidos los vectores, usamos Python para comparar qué tan cerca están unas palabras de otras. En este ejemplo comparamos "perro" con "fiel" y con "casa".
```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

embedding_perro = np.array([0.0087656295, 0.014512759, ...])
embedding_fiel  = np.array([-0.008843832, -0.0060662692, ...])
embedding_casa  = np.array([0.017014313, -0.004602901, ...])

print("Similitud Perro vs Fiel:", cosine_similarity(embedding_perro, embedding_fiel))
print("Similitud Perro vs Casa:", cosine_similarity(embedding_perro, embedding_casa))
` `` `

## Conceptos Clave: cómo funciona por debajo

### ¿Qué es un Embedding?

Imagina que cada palabra es un punto en un mapa gigante. En lugar de tener solo 2 dimensiones (norte-sur, este-oeste), tiene miles. Las palabras con significados similares, como "perro" y "fiel", terminan ubicadas muy cerca una de la otra en ese mapa.

### Similitud de Coseno

Para saber qué tan parecidas son dos palabras, no medimos la distancia en línea recta sino el ángulo entre sus vectores. La fórmula es:
```
similitud(A, B) = (A · B) / (||A|| * ||B||)
` `` `

- Si el resultado es **1**: el ángulo es 0°, las palabras son idénticas o muy similares.
- Si el resultado es **0**: son ortogonales, no tienen relación semántica.
- Si el resultado es **-1**: son opuestos totales.

### Por qué usamos Coseno y no distancia euclidiana

En IA nos importa más la dirección del concepto que la magnitud del vector. Por ejemplo, "perro" y "perros" pueden tener magnitudes distintas por su frecuencia, pero su dirección en el espacio de conceptos es casi la misma.

## Conclusión

Al ejecutar este código verás que la similitud entre "perro" y "fiel" es mayor que entre "perro" y "casa". Esto demuestra que el modelo ha aprendido que existe una relación conceptual y cultural entre los canes y la fidelidad, mostrando que la IA no solo lee letras sino que entiende relaciones.
````
