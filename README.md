# Ticket Intelligence API

Pipeline de clasificación de tickets de soporte en español con **FastAPI,
scikit-learn y Docker**. Cubre el ciclo completo —datos → modelo → API →
contenedor— con foco en reproducibilidad, evaluación y calidad de software.

El modelo asigna una de cinco categorías: `facturacion`, `acceso`,
`soporte_tecnico`, `ventas` y `cancelaciones`. Devuelve probabilidades y
marca `needs_review` si la mayor probabilidad es menor al umbral configurado
(0.60 por defecto).

> **Sobre el dataset:** los 80 ejemplos incluidos son sintéticos y tienen
> solo fines de demostración. Las métricas del proyecto verifican que el
> pipeline funciona, no miden desempeño real. Para resultados
> representativos se requiere un dataset real anonimizado; ver
> [Roadmap](#roadmap).

## Capturas

> _Pendiente: agregar un GIF de `/docs` mostrando una llamada a `/predict`
> con `category`, `confidence` y `needs_review`._

## Inicio local

Requiere Python 3.13.

### Windows / PowerShell

Si `python` abre Microsoft Store, usa la ruta de tu instalación o `py -3.13`
para crear el entorno.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m training.train
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

### Linux / macOS / WSL

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m training.train
uvicorn app.main:app --reload
```

Abre http://localhost:8000/docs para probar la API desde Swagger UI. No
necesitas activar el entorno virtual si usas la ruta explícita al intérprete.

## Docker

Inicia Docker Desktop con contenedores Linux y ejecuta:

```powershell
docker compose up --build
```

El contenedor entrena el modelo durante la construcción para que la imagen
sea autocontenida; en un entorno de producción el artefacto se inyectaría
ya entrenado, no se generaría en el build. La API se ejecuta con un usuario
sin privilegios y Compose publica el puerto solamente en la máquina local.
Para detenerlo: `docker compose down`.

## Ejemplo

```powershell
$body = @{ text = 'Me cobraron dos veces la suscripción' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

La respuesta contiene `category`, `confidence`, `needs_review`,
`probabilities` y `model_version`. `confidence` es la probabilidad estimada
por el clasificador; no es una garantía de acierto ni una probabilidad
calibrada.

| Método | Ruta | Uso |
| --- | --- | --- |
| GET | `/health` | Estado del proceso con modelo cargado |
| GET | `/model` | Versión, evaluación y umbral |
| POST | `/predict` | Un ticket: `{"text":"..."}` |
| POST | `/predict/batch` | De 1 a 100 tickets: `{"tickets":[{"text":"..."}]}` |
| GET | `/docs` | Documentación interactiva |

Los textos deben tener entre 5 y 5000 caracteres después de eliminar
espacios exteriores. Los datos inválidos producen HTTP 422. La API falla al
iniciar si falta el artefacto, en lugar de entrenar silenciosamente al
recibir una petición.

## Modelo y evaluación

```text
CSV → partición estratificada 75/25 → TF-IDF → {LogReg, MultinomialNB}
                                          ↓
                            mejor modelo por F1 macro
                                          ↓
                            modelo + métricas → FastAPI
```

- `data/tickets.csv`: 80 ejemplos sintéticos escritos para esta demo; 16
  por categoría. No contienen tickets de clientes reales.
- `training/train.py`: semilla 42, separación antes de ajustar TF-IDF y
  entrenamiento sin usar el conjunto de prueba.
- `artifacts/metrics.json`: accuracy, F1 macro, comparación con dos
  baselines (clase mayoritaria y MultinomialNB sobre el mismo TF-IDF),
  reporte por clase y matriz de confusión. Filas = clase real; columnas =
  clase predicha en el orden de `categories`.
- `artifacts/model.joblib`: pipeline evaluado y metadatos; se genera
  localmente y no se versiona. Carga únicamente artefactos propios y
  confiables: joblib usa deserialización basada en pickle.

Las métricas de este conjunto pequeño y sintético solo verifican el
funcionamiento del proyecto. No representan desempeño en producción. Hay
similitud de vocabulario entre ejemplos, no hay tickets ambiguos ni
evaluación fuera de dominio. El modelo siempre elige una de las cinco
clases, incluso para textos ajenos al soporte; `needs_review` no resuelve
por sí solo ese problema.

`needs_review` depende de que `confidence` sea una probabilidad calibrada;
con el modelo actual la confianza no está calibrada, así que el umbral de
0.60 debe interpretarse como una heurística, no como una probabilidad real
de error.

Para usar otros datos, conserva las columnas `text,category` y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m training.train --data data/tickets.csv --output artifacts
```

Usa suficientes ejemplos por clase para la partición estratificada. Con
datos reales, anonimiza información personal, separa duplicados y
conversaciones relacionadas antes de dividir, y reserva una evaluación
temporal independiente. Cambia `model_version` al producir una nueva versión
del modelo.

## Verificación

Requiere las dependencias de `requirements-dev.txt` (`pytest`, `ruff`).

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Las pruebas verifican contratos HTTP, clasificación de ejemplo, validación
de entrada, lotes, revisión manual, falta de modelo y entrenamiento
reproducible. GitHub Actions ejecuta lint, pruebas y construcción Docker al
subir el repositorio. La ejecución remota requiere publicar el repositorio y
habilitar Actions.

Configuración: `MODEL_PATH` permite indicar otro artefacto confiable;
`REVIEW_THRESHOLD` admite valores de 0 a 1. Consulta `.env.example` para los
valores por defecto. La respuesta incluye el tiempo de procesamiento y los
logs evitan registrar el contenido del ticket. Esta demo local no incluye
autenticación, limitación de solicitudes ni monitoreo de deriva.

## Estructura

```text
app/                # FastAPI: rutas, esquemas Pydantic y carga del modelo
training/           # Entrenamiento reproducible y evaluación
data/               # Dataset sintético (tickets.csv)
artifacts/          # Modelo y métricas generados localmente (no versionados)
tests/              # Pruebas con pytest
.github/workflows/  # CI: lint, tests y build Docker
```

## Roadmap

- Dataset real anonimizado y análisis de errores por clase.
- Comparación con embeddings (Sentence Transformers).
- Calibración de probabilidades y validación del umbral.
- Despliegue con autenticación y limitación de solicitudes.
- Monitoreo básico de deriva y logs estructurados.

## Licencia

MIT. Ver [LICENSE](LICENSE).

## Referencias

- [FastAPI: contenedores Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [scikit-learn: persistencia y seguridad de modelos](https://scikit-learn.org/stable/model_persistence.html)