
</think>

# Comprender y Corregir el Razonamiento Multietapa Composicional en Grandes Modelos de Lenguaje
Este repositorio contiene la implementación oficial del artículo [*Comprender y Corregir el Razonamiento Composicional en LLMs*](https://arxiv.org/abs/2402.14328) (ACL'2024, Findings)

## Instalar el entorno de conda y los paquetes
Los archivos yaml del entorno se encuentran en el directorio `./environments`. Tenga en cuenta que disponemos de dos entornos: uno para experimentos de investigación (inferencia, logit lens, intervención causal y localización) y otro para experimentos de parcheo (creme).
- Para experimentos de investigación: ejecute el comando `conda env create -n investigate environments/investigating/environment.yaml`; luego active el entorno `conda activate investigate` al ejecutar los experimentos de investigación.
- Para experimentos de parcheo: ejecute el comando `conda env create -n patch environments/patching/environment.yaml`; luego active el entorno `conda activate patch` al ejecutar los experimentos de parcheo.
## Datos y Modelos
Para los datos, cambie la ruta de trabajo al directorio `./data` y ejecute el comando `cd mquake`.
- Los datos originales de MQuAKE-CF (división de 2 saltos): `MQuAKE-CF-3k.2hop.json`. Este archivo de datos se utiliza directamente para los experimentos de inferencia (brecha de composicionalidad).
- Para los experimentos de inferencia, con el fin de alinear el formato de salida de los LLMs al espacio de respuestas (es decir, que devuelvan directamente la respuesta en lugar de otras palabras comunes (p. ej., "OK", "Genial", repetir la pregunta, etc.)), utilizamos prompts few-shot para instruir a los modelos. Las plantillas se pueden encontrar en `./data/mquake/prompts`.
- Para los experimentos de inspección, causalidad y localización: utilizamos `comp_cloze_prefix.json` (o `comp_cloze_suffix.json`), donde parafraseamos los elementos de conocimiento en forma de prueba de hueco (Cloze-Test) (es decir, (sujeto, relación, objeto): El creador de C. Auguste Dupin es __ (esperando completar), de acuerdo con trabajos previos, p. ej., [ROME](https://arxiv.org/abs/2202.05262), [Memory Injections](https://arxiv.org/abs/2309.05605), [Dissecting Factual Recall](https://arxiv.org/abs/2304.14767) y cetera).
- Para los experimentos de edición (parcheo): construimos `MQuAKE-CF-3k.2hop.edit.json`, donde muestreamos conjuntos de parafraseo, generalización e irrelevancia (consulte el artículo para una introducción detallada) para cada caso de prueba sobre la base de `MQuAKE-CF-3k.2hop.json`. Tenga en cuenta que en `MQuAKE-CF-3k.2hop.edit.json`, los casos de prueba irrelevantes podrían ser ruidosos (comparten la respuesta con el caso que se va a parchear). Por lo tanto, re-muestreamos los casos irrelevantes en `./creme/make_dataset/make_dataset_irrelevant.py`.

## Experimentos de Inferencia (Brecha de Composicionalidad)
Para ejecutar los experimentos de inferencia (Brecha de Composicionalidad, Errores de Razonamiento Composicional), cambie la ruta de trabajo al directorio `inference` (`cd inference/MQuAKE`).
- Para ejecutar la inferencia para preguntas de un solo salto, ejecute `python inference_single.py <model_name>`, donde `<model_name>` puede ser `llama2-7b`, `llama2-13b` o `openalpace-3b`. Después de finalizar la ejecución del programa de inferencia, se guardará automáticamente un archivo de resultados (`<model_name>.json`) en el directorio `inference/MQuAKE/single-hop`.
- Para ejecutar la inferencia para preguntas composicones de dos saltos, ejecute `python inference_comp.py <model_name>`. Después de finalizar la ejecución del programa de inferencia, se guardará automáticamente un archivo de resultados (`<model_name>.json`) en el directorio `inference/MQuAKE/compositional`.
- Después de obtener los resultados de inferencia tanto para preguntas de un solo salto como para preguntas composicones de dos saltos, podemos ejecutar `python filter.py <model_name> <fix_type>` para clasificar los resultados en dos categorías: (1) `pass_all`, lo que significa que el LLM puede responder correctamente tanto a las preguntas de un solo salto como a las composicones correspondientes; (2) `pass_singles_fail_comp`, lo que significa que el LLM, aunque responde correctamente las preguntas de un solo salto, falla en resolver las composicones (en relación con la [Brecha de Composicionalidad](https://aclanthology.org/2023.findings-emnlp.378/), Errores de Razonamiento Composicional). Ambas partes de los resultados se guardarán por separado en dos archivos en el directorio `inference/MQuAKE/filter`.
- Notas: `<fix_type>` puede ser `prefix` o `suffix`, indicando dos órdenes diferentes al componer dos preguntas de un solo salto. Esto es para uso futuro. Además, en cada caso de prueba individual, hay tres preguntas composicones parafraseadas (que comparten el mismo significado) para probar el modelo. Siguiendo el artículo original de [MQuAKE](https://arxiv.org/abs/2305.14795), consideramos que el modelo pasa la prueba siempre que pueda responder correctamente una de las tres preguntas parafraseadas.

## Experimentos de Logit Lens
Las tres siguientes partes (*inspección de logit lens, experimentos de intervención y experimentos de localización*) se encuentran en el directorio `inspecting_and_intervention`, las cuales fueron implementadas principalmente sobre la base de [la implementación oficial de ROME](https://github.com/kmeng01/rome) (¡esto es un reconocimiento!).
- Para ejecutar ejemplos de Logit Lens, cambie la ruta de trabajo: `cd inspecting_and_intervention` y ejecute el programa: `python logit_lens.py`. Tenga en cuenta que el ejemplo de prueba está codificado de forma rígida en el programa (por lo que necesitamos modificarlo manualmente para probar diferentes casos). Una ejecución exitosa del programa generará una figura de la curva de logit lens en el directorio `inspecting_and_intervention/logit_lens/results`.
## Experimentos de Intervención
Para ejecutar los experimentos de intervención causal, primero cambie la ruta de trabajo a `cd inspecting_and_intervention/causal_intervention`.
- Primero, obtenga los datos de intervención causal: ejecute el comando `python fetch.py <fix_type> <model_name>`, donde `<fix_type>` puede ser `prefix` o `suffix`; `<model_name>` puede ser `llama2-7b` o `openalpaca-3b`. Este programa obtendrá los datos de intervención y los organizará en un archivo `<model_name>.<fix_type>.json` en el directorio actual.
- Para ejecutar el experimento de intervención causal: ejecute el comando `python causality.py <model_name> <fix_type>`. Este programa generará un archivo de resultados `<model_name>.<fix_type>.json` en el directorio `results`.
- Para agregar los resultados (promedio sobre instancias) y visualizarlos: primero cambie la ruta de trabajo al directorio `results` (`cd results`) y ejecute el comando `python aggregate_visualize.py <model_name> <fix_type>`. Una ejecución exitosa generará una figura de mapa de calor en el mismo directorio.
## Experimentos de Localización
Para ejecutar los experimentos de localización, primero cambie la ruta de trabajo a `cd inspecting_and_intervention/locating`.
- Para ejecutar los experimentos de localización, ejecute el comando `python locating.py <model_name> <fix_type>`. Una ejecución exitosa del programa generará un archivo de resultados `<model_name>.<fix_type>.json` en el directorio `inspecting_and_intervention/locating/results`.
- Para agregar los resultados (promedio sobre instancias) y visualizarlos: primero cambie la ruta de trabajo al directorio `results` (`cd results`) y ejecute el comando `python aggregate_visualize.py <model_name> <fix_type>`. Una ejecución exitosa generará una figura de mapa de calor en el mismo directorio.
## Corrección de Errores de Razonamiento Composicional mediante Edición de Modelos
Para ejecutar los experimentos de parcheo, primero cambie la ruta de trabajo a `cd creme`. Esta parte del código fue construida sobre la base de [FastEdit](https://github.com/hiyouga/FastEdit).
- Para preparar los datos de edición, primero `cd make_dataset`. 
  - Para las pruebas de *corrección*, *parafraseo* y *generalización*, ejecute `python make_dataset.py <model_name>` (`llama2-7b` o `openalpaca-3b`).
  - Para las pruebas de *irrelevancia*, ejecute `python make_dataset_irre.py <model_name>`.
  - Después de generar los datos de edición en la ruta del directorio actual, regrese a la carpeta anterior `cd ..`.
- Para obtener resultados estadísticos, primero `cd fastedit_comp`. Ejecute el comando `bash test_batch.sh` para obtener los resultados de las pruebas de *corrección*, *parafraseo* y *generalización*. Ejecute el comando `bash test_batch_irre.sh` para obtener los resultados de las pruebas de *irrelevancia*.
  - Los resultados se pueden encontrar en `results/v0` (para pruebas no irrelevantes) o `results/irrelevant` (para pruebas irrelevantes). Ejecutar `python results/aggregate.py <testing_type>` (testing_type = v0 o irrelevant) generará los resultados promediados.
- Para probar un caso único,
  - Prepare el caso de prueba en `creme/data`, siguiendo el formato de `example.json` (el caso de nacionalidad, creador, C. Auguste Dupin).
  - Cambie la ruta de trabajo a `cd fastedit_comp` y luego ejecute el comando `test.sh`. El contenido de la salida se puede ver en el archivo `testing.txt`.

## Citación
Si le ha útil el artículo o el repositorio, le agradeceríamos mucho que considerara citar el artículo (con el siguiente bibtex):
```
@inproceedings{li-etal-2024-understanding,
    title = "Understanding and Patching Compositional Reasoning in {LLM}s",
    author = "Li, Zhaoyi  and
      Jiang, Gangwei  and
      Xie, Hong  and
      Song, Linqi  and
      Lian, Defu  and
      Wei, Ying",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.576",
    pages = "9668--9688",
    abstract = "LLMs have marked a revolutonary shift, yet they falter when faced with compositional reasoning tasks. Our research embarks on a quest to uncover the root causes of compositional reasoning failures of LLMs, uncovering that most of them stem from the improperly generated or leveraged implicit reasoning results. Inspired by our empirical findings, we resort to Logit Lens and an intervention experiment to dissect the inner hidden states of LLMs. This deep dive reveals that implicit reasoning results indeed surface within middle layers and play a causative role in shaping the final explicit reasoning results. Our exploration further locates multi-head self-attention (MHSA) modules within these layers, which emerge as the linchpins in accurate generation and leveraing of implicit reasoning results. Grounded on the above findings, we develop CREME, a lightweight method to patch errors in compositional reasoning via editing the located MHSA modules. Our empirical evidence stands testament to CREME{'}s effectiveness, paving the way for autonomously and continuously enhancing compositional reasoning capabilities in language models.",
}
```
## Poster
![Understanding and patching Compositional Reasoning in LLMs](poster_acl2024_creme_page-0001.jpg)
