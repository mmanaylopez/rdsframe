# Auditoría técnica y evolución de `rds_reader.py`

## Diagnóstico del original

El archivo original resuelve un problema real: evita construir el árbol completo
de objetos de un lector R generalista y convierte vectores atómicos a NumPy. Sin
embargo, todavía tenía cuatro cuellos de botella para archivos grandes:

1. `_raw(4 * n)` y `_raw(8 * n)` creaban un bloque `bytes` del tamaño de la
   columna y luego `astype()` creaba otro array completo.
2. `read_rds_dataframe()` convertía primero a Parquet, volvía a leer el Parquet y
   limpiaba archivos manualmente.
3. `convert_rds()` reunía todas las columnas en un DataFrame antes de escribir,
   por lo que el pico de RAM seguía aproximándose al tamaño completo de la tabla.
4. Las cadenas se almacenaban en una lista de referencias a objetos Python; el
   interning ayudaba con baja cardinalidad, pero no con texto casi único.

También faltaban una API tipada, excepciones diferenciadas, límites defensivos,
escritura atómica, CLI, pruebas, metadatos modernos y automatización de CI.

## Arquitectura implementada

- `Reader` llena arrays NumPy nativos mediante `readinto()` y realiza el cambio
  de endianess in-place. Se elimina el buffer `bytes` completo intermedio.
- `read_rds()` conserva la ruta pandas para tablas que caben en memoria.
- `to_parquet()` es la ruta de gran volumen:
  1. lee una columna del flujo RDS;
  2. construye una representación Arrow;
  3. acumula un lote acotado por número de columnas y bytes Arrow;
  4. guarda el lote temporal en Parquet y lo libera;
  5. une posicionalmente los lotes y genera el Parquet final de forma atómica.
- Los `STRSXP` se construyen con buffers Arrow `LargeString`: offsets `int64`,
  datos UTF-8 y bitmap de validez. No se conserva un objeto `str` por fila.
- DuckDB recibe `memory_limit` y `temp_directory`, por lo que puede derramar el
  trabajo de consolidación a disco.

## Qué significa “streaming” aquí

R serializa un `data.frame` por columnas y coloca sus nombres al final. Por ello,
la unidad mínima razonable de materialización es una columna, no una fila. El
pico del parser queda ligado principalmente a la columna individual más grande.
No es correcto prometer una cifra universal como “menos de 1 GB”: depende del
tipo, longitud, cardinalidad del texto, compresión y configuración de DuckDB.

El GC cíclico explícito se ejecuta por intervalo configurable, no por columna.
`gc_collect_every=0` lo desactiva; la liberación normal de NumPy/Arrow continúa
mediante conteo de referencias.

Una lista cuyo primer elemento sea otro `data.frame` presenta una ambigüedad con
una columna-lista. En ese caso, el primer `data.frame` se materializa para
clasificar la estructura; las tablas siguientes vuelven al procesamiento por
columnas. Esta limitación está documentada y debe medirse con corpus reales.

## Escritura RDS y RData

Es técnicamente posible, pero no forma parte de la versión 0.1.0. Un escritor
debe conservar exactamente tipos R, atributos, factores, zonas horarias,
codificaciones, `NA` frente a `NaN`, referencias y versiones de serialización.

La opción pragmática es un extra opcional basado en `pyreadr`, que ofrece
`write_rds()` y `write_rdata()` para un único DataFrame. Sus limitaciones
declaradas incluyen un solo objeto, ausencia de row names, diferencias entre
`NA` y `NaN`, velocidad y soporte restringido de tipos. Antes de integrarlo hay
que revisar el efecto de sus licencias y probar archivos generados contra R.

Para una implementación propia se recomienda:

1. empezar por RDS XDR v3, `data.frame` único y tipos atómicos;
2. generar fixtures en R y hacer pruebas bidireccionales Python → R → Python;
3. agregar factores, fechas, POSIXct, codificaciones y compresión;
4. abordar RData y varios objetos solo después de estabilizar RDS;
5. ejecutar fuzzing y pruebas de corrupción/truncamiento.

## Validación y publicación

La distribución incluye pruebas XDR y gzip, nulos, factor, fecha, lógico, texto
UTF-8, cabeceras, truncamiento, CLI y consulta DuckDB del Parquet. La matriz CI
cubre Linux, Windows y macOS con Python 3.10–3.13. Antes de publicar se debe:

- confirmar o cambiar el nombre de distribución `rdsframe` en PyPI;
- actualizar las URLs si el repositorio final es distinto;
- probar con un corpus de RDS producido por varias versiones de R;
- ejecutar benchmarks de tiempo, RSS pico y disco temporal contra la versión
  original y lectores alternativos;
- publicar primero en TestPyPI y validar instalación limpia.

## Evolución 0.3.0a1

Una implementación adaptada para una aplicación propuso truncar listas de
`data.frame`, eliminar siempre la zona horaria de `POSIXct` y convertir
columnas-lista incompatibles mediante `str()`. La versión pública no adopta
esas decisiones silenciosas:

- los límites generan `RDSLimitError` antes de producir resultados parciales;
- `POSIXct` conserva la zona por defecto y permite optar por UTC naive;
- timestamps infinitos o fuera de rango fallan salvo coerción explícita;
- columnas-lista heterogéneas fallan o se convierten mediante una política
  elegida (`json` determinista o `string`);
- complejos se representan como `STRUCT(real, imag)` en Parquet y los vectores
  raw como `uint8`.

## Evolución 0.4.0a1: catálogo y selección

El formato RDS es secuencial y los nombres de una lista suelen aparecer como
atributo después de todos sus elementos. Por ello no existe acceso aleatorio
gratuito en un RDS comprimido. La nueva arquitectura separa dos costos:

- `list_rds_tables()` atraviesa el flujo, pero descarta los grandes buffers en
  bloques acotados y solo materializa atributos pequeños;
- en RDS sin compresión, los payloads de tamaño conocido se saltan directamente
  con `seek()` y validación contra el tamaño del archivo;
- `extract_rds_tables()` convierte únicamente los índices elegidos;
- un catálogo validado permite resolver nombres y detener la segunda pasada al
  terminar la última tabla seleccionada;
- sin catálogo, seleccionar por nombre requiere primero un escaneo estructural;
- el catálogo se invalida si cambian ruta, tamaño o `mtime_ns` del archivo.

Las tablas omitidas no llegan a pandas, Arrow, DuckDB ni a temporales Parquet.

## Evolución 0.4.0a2: el cuello de botella real era el texto

Medido contra un RDS real de producción (123 MiB, gzip, 6 `data.frame`, ~5.5
millones de filas totales, con muchas columnas de texto de valores muy
repetidos), `list_rds_tables()` tardaba **165.4 s**. El perfil de CPU confirma lo que ya sugería la arquitectura: el
costo no está en `gzip`, está en `char()`/`skip_char()`, invocados una vez por
cada elemento `CHARSXP` — decenas de millones de veces en tablas anchas con
texto. Cada invocación pagaba, antes de esta versión:

1. `flags()` construía una tupla de 5 elementos (3 `bool()`) que estas rutas
   nunca leen; solo usan `sexp_type` y los bits crudos.
2. `i32()` recompilaba un `struct.Struct` implícito por llamada (el caché
   interno de `struct` evita recompilar el formato, pero no evita el costo de
   formatear el string `f"{byteorder}i"` en cada llamada).
3. `raw()`/`discard()` añadían una capa de método completa por lectura, y
   `discard()` reconstruía un `memoryview` del buffer compartido en cada
   invocación en vez de reutilizar uno cacheado.
4. `discard()` llamaba a `tick()` en cada bloque leído incluso sin callback de
   progreso configurado (el caso común), pagando una llamada de función que
   no hacía nada.
5. Los streams comprimidos (`gzip.GzipFile`, `bz2.BZ2File`, `lzma.LZMAFile`)
   exponen un `io.BufferedReader` interno de 8 KiB no configurable; miles de
   lecturas de 4-8 bytes golpeaban ese límite en vez de amortizarse en un
   buffer mayor.

La corrección fue mecánica y preserva el comportamiento exacto: métodos
cacheados (`self._read`, `self._readinto`) y un `struct.Struct` precompilado
por `Reader`, cabeceras de `CHARSXP` leídas en línea sin la tupla de
`flags()`, un camino rápido de una sola lectura en `discard()` para el caso
común (un string cabe en el buffer compartido), `tick()` evitado por completo
cuando no hay callback, y los tres formatos comprimidos envueltos en un
`io.BufferedReader` externo de 1 MiB (probar un buffer casero propio para
evitar la comprobación `raw.closed` de esa clase resultó, medido, más lento
que `io.BufferedReader`, así que se descartó). Resultado, mismo archivo,
mismo catálogo de salida: **165.4 s → 66.7 s** (~2.5×). Ningún test cambió de
comportamiento; la suite completa (52/52, incluidos los casos que requieren
DuckDB) sigue en verde, y `to_parquet()` se validó extremo a extremo contra
el mismo archivo real (589 554 y 420 filas convertidas y verificadas con
`duckdb.sql(...)` contra el Parquet resultante).

Adicionalmente, `Reader.char_utf8()` evita el viaje decode→encode que hacía
`arrow_string_array()`: R marca con sus propios bits `gp` cuándo un string ya
es UTF-8 o ASCII puro; en ese caso (la mayoría en la práctica) se usan los
bytes crudos directamente, sin construir nunca un `str` intermedio. Solo el
texto genuinamente latin-1 o sin bandera de codificación explícita paga la
transcodificación real.

## Evolución 0.4.0a2: lectura selectiva de columnas

`to_parquet`/`extract_rds_tables` ya evitaban materializar tablas no
seleccionadas; `read_rds()` no tenía equivalente a nivel de columna dentro de
un único `data.frame`. `read_rds(path, columns=[...])` reutiliza el mismo
principio de `skip_item()` estructural: las columnas no pedidas nunca se
convierten a NumPy/pandas. La selección por nombre requiere un pase
estructural previo (igual que la selección de tablas por nombre) porque R
escribe los nombres de columna después de todas las columnas; la selección
por índice entero evita ese pase. La funcionalidad está acotada
deliberadamente a un único `data.frame` raíz — un RDS con varias tablas debe
resolverse primero con `extract_rds_tables`/`to_parquet`.

**El ahorro real depende del tipo de las columnas omitidas, no es "gratis"
para cualquier tabla.** RDS es un formato secuencial de longitud variable:
omitir una columna numérica/lógica/raw de un RDS **sin comprimir** es un
`seek()` real (`discard()` toma la ruta `seekable_discard`, coste
prácticamente nulo). Omitir una columna de texto (`STRSXP`) exige recorrer
igual un `skip_char()` por fila — el mismo número de operaciones que leerla,
solo evitando la decodificación y el `pandas.Series`/interning resultante.
En un RDS **comprimido**, además, hay que descomprimir los bytes de la
columna omitida de todos modos para llegar a la siguiente. Medido con una
tabla sintética sin comprimir de 300 000 filas y 30 columnas (2 de texto, 28
numéricas): pedir 3 columnas por índice tomó 0.74 s frente a 0.79 s de leer
las 30 — una mejora real pero modesta, no un salto de orden de magnitud,
porque la mayoría de las columnas omitidas ya eran baratas de leer.
`columns=` sigue siendo valioso principalmente por **RAM** (las columnas
omitidas nunca llegan a ocupar un array NumPy ni objetos `str` internados) y
por CPU en tablas muy anchas con muchas columnas numéricas no deseadas; no
es una promesa de "columnas de texto gratis".

Esta iteración también corrigió un error real detectado durante la
validación: el `Reader` interno de `_read_dataframe_selective` no propagaba
`seekable_discard=stream is raw`, por lo que la ruta de `seek()` rápido
nunca se activaba para RDS sin comprimir, incluso cuando el archivo lo
permitía. Corregido antes de la primera publicación.

## Evolución 0.4.0a3: ampliar la validación más allá de un solo archivo

Antes de publicar en PyPI se reunieron 6 archivos RDS reales adicionales
(fuera del archivo de producción original) específicamente para no
sobreajustar las decisiones de diseño a un solo caso: una lista plana con
nombres de 24 206 entradas (versión de serialización **2**, no un
`data.frame`), un snapshot de metadatos con 27 315 tablas, un dataset de
186 MiB sin comprimir (557 691 filas × 33 columnas), dos archivos de
resultados generados por herramientas, y un objeto de lista anidada plana
sin nombres ni clase. Todos se comportan como se espera: los que son
`data.frame` (o listas de ellos) se leen correctamente; los que no lo son
fallan con el mismo error claro y explícito de siempre.
Este corpus reveló tres defectos reales de conversión de tipos, no de
rendimiento:

1. **Factores ordenados perdían el orden en silencio.** El código solo
   comprobaba `"factor" in classes`, nunca `"ordered"`, así que
   `pd.Categorical.from_codes(...)` siempre se construía con
   `ordered=False`. Corregido en `api.py` (ruta pandas) y `_parquet.py`
   (`pa.DictionaryArray.from_arrays(..., ordered=...)`, ruta Arrow).
2. **`difftime` se convertía en `float64` plano, descartando el atributo
   `units`** sin aviso. Ahora se convierte a `pandas.Timedelta`
   (`pa.duration("us")` en Parquet) según la unidad real (`secs`, `mins`,
   `hours`, `days`, `weeks`), con `UnsupportedRDS` explícito si apareciera
   una unidad no reconocida — en vez de asumir segundos silenciosamente.
3. **Una columna de `data.frame` con atributo `dim`** (una matriz de R
   almacenada como una sola columna) producía un error confuso ("columnas
   de distinta longitud") en el caso general, y **pasaba sin ningún error
   en el caso `dim = c(n, 1)`**, ya que una matriz de una sola columna
   tiene exactamente `n` elementos — el mismo conteo que las demás columnas
   del `data.frame`. Este último caso era una aceptación silenciosa de una
   estructura incorrecta, no solo un mensaje de error pobre. Ahora
   `_column_to_pandas`/`_column_to_arrow` comprueban `attributes.get("dim")`
   explícitamente antes de cualquier otra lógica y lanzan
   `UnsupportedRDS("matrix- or array-valued data.frame columns are not
   supported")`.

Al verificar el fix de factores ordenados contra el pipeline completo de
`to_parquet()` se descubrió (no se introdujo) una limitación preexistente:
DuckDB, al escribir el Parquet final, ya materializaba **cualquier** columna
de diccionario/categórica de Arrow como cadena de texto plana — esto ocurre
igual para factores sin ordenar, con o sin este cambio. `_column_to_arrow()`
sí construye correctamente el `DictionaryArray` con `ordered=True` (probado
directamente); lo que no sobrevive es la tipificación categórica a través
del paso de staging de DuckDB, no algo que este fix haya afectado. Los
valores en el Parquet final siguen siendo correctos.

## Evolución 0.4.0a4: `read_r_object()` — más allá del data.frame

Durante la validación contra la lista de nombres de 24 206 entradas surgió
una pregunta directa: ese archivo (y otro de lista anidada) genuinamente no
son `data.frame`, así que `read_rds()` los rechaza correctamente — pero eso
deja al usuario sin forma de acceder a su contenido en absoluto. Dado que el objetivo declarado es
publicar en PyPI para ayudar a cualquiera con sus archivos RDS (no solo con
data.frames), y que el propio `Reader.read_item()` ya parsea genéricamente
`NILSXP`/`LGLSXP`/`INTSXP`/`REALSXP`/`CPLXSXP`/`STRSXP`/`RAWSXP`/`VECSXP`
sin exigir que la raíz sea tabular, la pieza que faltaba no era capacidad de
parseo — era una función pública que expusiera esa estructura como tipos
nativos de Python en vez de forzar todo a un `DataFrame`.

`read_r_object()` recorre recursivamente el árbol ya parseado: un `VECSXP`
con atributo `names` se vuelve `dict`; sin nombres, `list`; un `VECSXP` con
`class="data.frame"` en cualquier nivel de anidamiento sigue convirtiéndose
en un `pandas.DataFrame` real (reutilizando `_is_dataframe`/`_to_dataframe`
sin cambios); los vectores atómicos reutilizan `_column_to_pandas` (mismo
manejo de NA, factor, `Date`, `POSIXct`, `difftime` que una columna de
data.frame) pero se desenvuelven a escalar u objeto Python plano en vez de
quedar como `pandas.Series`. Las columnas-matriz (`dim`), que
`_column_to_pandas` ahora rechaza explícitamente para el caso data.frame,
aquí SÍ se soportan: se reconstruyen con `numpy.reshape(..., order="F")`
respetando el orden column-major de R.

Validación contra los dos archivos reales que motivaron esto:

- La lista plana con nombres (24 206 entradas, versión 2): ahora se lee como
  `dict[str, dict[str, str]]` en 29 s. Antes: `UnsupportedRDS` inmediato,
  sin ninguna forma de inspeccionar el contenido.
- El objeto de lista anidada: se lee como una lista anidada de 10×10 en
  0.28 s — y esa lectura reveló que contiene **data.frames reales anidados**
  que la inspección estructural superficial de la iteración 0.4.0a3 no había
  detectado, al limitarse a los dos primeros niveles de anidamiento sin
  recursar más profundo.

Los mensajes de error de `read_rds()`/`list_rds_tables()` cuando la raíz no
es un `data.frame` ahora mencionan `read_r_object()` explícitamente como
alternativa, en vez de dejar al usuario a adivinar.

`read_r_object()` es deliberadamente el camino general/exploratorio, no el
optimizado: materializa toda la estructura en memoria (como `read_rds()`),
sin streaming ni límites de columnas. Para archivos genuinamente tabulares,
`read_rds()`/`to_parquet()` siguen siendo la ruta recomendada.

## Evolución 0.4.0a5: validación contra R real y cierre de las limitaciones

Al descubrir R 4.5.0 instalado en la máquina de desarrollo, se generó un
corpus de 17 archivos RDS escritos por R mismo (`tests/data/gen_fixtures.R`,
persistido como archivos dorados en `tests/data/r450`, ~4.8 KiB en total).
El resultado inmediato invalidó una suposición central de las iteraciones
anteriores: **el soporte ALTREP existente nunca había funcionado contra
archivos reales de R moderno**. Los fixtures sintéticos construidos byte a
byte no serializaban ALTREP, así que el código nunca se había ejercitado de
verdad. Fallaba hasta `data.frame(x = 1:3)`, porque `1:n` se serializa como
`compact_intseq`.

Dos causas de raíz, ambas estructurales:

1. El *info* del ALTREP es un pairlist propio pero **sin tags**
   (`(símbolo_de_clase símbolo_de_paquete tipo)`), y `read_pairlist()`
   descartaba las entradas sin tag, de modo que el nombre de clase se
   perdía y todo ALTREP terminaba en `"ALTREP class is not supported: ?"`.
2. El *estado* de los wrappers (`wrap_integer`, `sort()` etc.) y de
   `deferred_string` es un **par punteado** `CONS(payload . metadata)` cuyo
   CDR es un vector plano, no otra celda; tanto `read_pairlist()` como
   `skip_pairlist()` lo rechazaban con "pairlist tail is not a LISTSXP".

La reescritura introduce `read_cons_chain()` (tolera celdas sin tag y CDR
punteado) y un `read_altrep()` que devuelve un `SerializedObject` completo
tipado como el vector subyacente. Detalle crítico encontrado en el camino:
el slot de atributos del ALTREP se **descartaba**; es portador de datos
reales (un `sort()` sobre un factor envuelve los códigos enteros y mueve
`levels`/`class` a ese slot), así que descartarlo habría degradado un factor
a enteros pelados sin error. Ahora se fusiona en el objeto resultante.
`deferred_string` se expande formateando como lo hace `as.character()` de R
(`%.15g` para dobles, distinguiendo el patrón de bits de `NA_real_` del NaN
ordinario, que R imprime como "NaN").

Con el corpus en verde, esta versión cierra además todas las limitaciones
documentadas en 0.4.0a3/a4:

- **POSIXlt**: los componentes de reloj de pared se reconstruyen a
  `Timestamp` naive (ruta pandas y Parquet); componentes NA o fechas
  inválidas producen `NaT`, nunca una excepción ni el error confuso de
  "different lengths".
- **Row names**: los nombres de fila de texto se vuelven el índice de
  pandas; la codificación compacta por defecto de R (`c(NA, -n)`) sigue
  produciendo `RangeIndex`.
- **Endianness del formato nativo**: la cabecera se valida en el orden de
  bytes asumido y se reintenta en el opuesto (el campo de versión solo
  puede ser 1-3); un archivo cruzado de arquitectura ahora da lectura
  correcta o `InvalidRDS` claro, nunca números silenciosamente corruptos.
- **Encoding nativo**: el campo de codificación que la propia cabecera v3
  declara (y que se leía pero se ignoraba) ahora es el default real,
  con `encoding=` como override en toda la API y `--encoding` en la CLI.
- **Buffers en memoria**: `read_rds()`/`read_rds_dataframe()`/
  `read_r_object()` aceptan `bytes`/`bytearray`/`memoryview` y streams
  binarios seekables; el stream del llamador no se cierra.
- **CLI `dump`**: árbol truncado (`--max-items`/`--max-depth`) o JSON
  completo (`--json`) de cualquier objeto soportado, para inspeccionar un
  RDS desconocido desde la terminal antes de decidir cómo convertirlo.

## Evolución 0.4.0a6: parseo de strings por lotes y cierre de la revisión externa

Una revisión externa del código identificó correctamente que las columnas de
texto eran el cuello de botella restante (~67 s para listar el archivo real
de 123 MiB) y lo calificó de "estructural del formato RDS, no un defecto del
código". El diagnóstico es medio cierto: cada `CHARSXP` trae su propia
cabecera y eso obliga a recorrer elemento por elemento — pero pagar **tres
llamadas de stream por string** no es estructural, es implementación. La
solución: los vectores de strings ahora se parsean desde bloques de 1 MiB
con `struct.unpack_from` sobre un buffer local; el excedente leído de más se
aparca en un buffer pendiente que `raw()`/`read_into()`/`discard()` drenan
antes de tocar el stream. Cuando un payload no cabe en el bloque, el salto
restante delega en `discard()`, que en archivos sin comprimir sigue siendo
un `seek()` real. Resultados (mismo archivo, misma máquina):

| Operación | antes | después |
| --- | ---: | ---: |
| `list_rds_tables()` 123 MiB gzip | 66.7 s | **17.1 s** |
| `read_rds()` 186 MiB, 557k filas | 4.6 s | **3.0 s** |
| `to_parquet()` de una tabla de 589k filas con catálogo | — | **5.0 s** |

Acumulado desde 0.4.0a1: 165.4 s → 17.1 s (9.7×). Lo que queda es el bucle
Python puro sobre ~44 M cabeceras; el siguiente escalón sería paralelizar
columnas independientes entre procesos, que exige fuente seekable o
re-descompresión por columna y queda fuera de esta versión.

El resto de las observaciones de esa revisión también quedó atendido:

- **Flag UTF-8 mal etiquetado** (la ruta rápida devolvía bytes inválidos si
  el archivo declaraba UTF-8 pero traía latin-1): el override `encoding=`
  ahora también valida y re-decodifica los strings con bandera UTF-8 cuya
  carga no es UTF-8 válido. Sin override, la ruta de confianza sin
  validación se mantiene intacta (coste cero por defecto, documentado).
- **Tipos que caían al fallback lento "en silencio"**: S4 y environments ya
  no necesitan fallback — `read_r_object()` los lee directamente (S4 como
  dict de slots con `$r_class`; environments como dict de contenido, con
  alineación de referencias verificada cuando el mismo environment aparece
  dos veces y soporte de enclosures de namespace/paquete). Closures,
  llamadas de lenguaje, promesas y bytecode fallan con el **nombre del tipo
  R** en el mensaje, para que la aplicación pueda explicar al usuario por
  qué activa su lector alternativo.
- **Descompresión a temp para poder hacer seek**: `materialize_uncompressed()`
  hace esa descompresión única de forma atómica y documentada, en lugar de
  que cada aplicación lo reimplemente. Nota de medición: descomprimir NO es
  el coste dominante del listado (el gzip de 123 MiB se infla en ~2 s); el
  beneficio del seek aplica sobre todo al salto de payloads numéricos
  grandes y al acceso selectivo repetido.
- **`STRUCT(real, imag)` para complejos** y **`duration(µs)` para
  `difftime`**: son representaciones, no tipos nativos de Parquet, y así
  quedan documentadas en el README (valor exacto preservado; la unidad de
  display de difftime no sobrevive en el tipo). Las unidades de `difftime`
  reconocidas (`secs`/`mins`/`hours`/`days`/`weeks`) son el conjunto
  cerrado que R acepta en `units<-`; cualquier otra falla explícita.
- **Listas heterogéneas**: el fallo por defecto sin `list_column_mode`
  explícito es una decisión de diseño (no coercionar con pérdida en
  silencio), no fragilidad; queda razonado en el README.

## Evolución 0.4.0a7: regresión de memoria en la ruta Arrow (hallazgo externo)

Una segunda revisión externa detectó una regresión real introducida en
0.4.0a6, y su diagnóstico fue exacto: al unificar la ruta Arrow sobre el
parser por lotes, `arrow_string_array()` pasó de transmitir valor a valor
directo a los buffers (pico ≈ 1 columna, el diseño original de 0.4.0a5) a
materializar la columna completa como lista de objetos `bytes` de Python
más un dict de interning **antes** de armar los buffers — pico ≈ 2× el
texto de la columna más el overhead por objeto (~72 bytes por `bytes` en
CPython). En un subproceso de conversión con `memory_limit="1GB"`, una
columna de texto suficientemente grande podía reventar el límite.

Corrección upstream (equivalente a la aplicada downstream): los elementos
se siguen parseando con el lector por lotes, pero se **drenan a los buffers
de Arrow en fragmentos acotados** de 262 144 filas (`_STRING_CHUNK`, la
granularidad por defecto del row-group de Parquet). El pico vuelve a ≈ 1
columna + 1 fragmento acotado, la velocidad medida no cambia
(`to_parquet` de la tabla de 589k filas sigue en ~5 s), y la salida es
idéntica — verificado con tests de equivalencia contra la ruta de strings
objeto, de fronteras de fragmento (fragmentos diminutos parcheados, un
payload mayor que el buffer de lotes cayendo justo en el borde), y un test
comparativo con `tracemalloc` que fija el mecanismo: drenar acotado debe
ahorrar memoria frente a materializar la columna entera, sin presupuestos
absolutos frágiles.

El interning quedó restringido al modo de strings objeto, donde la lista
devuelta es el resultado longevo y deduplicar ahorra memoria real; en el
modo Arrow los fragmentos son transitorios y el lookup por elemento era
overhead puro.

Del resto de observaciones de esa revisión: el try/except no-op de
`read_item_from_header()` se eliminó (aquí no hay "upstream del que no
divergir": esto ES upstream). La lectura completa del enclosure en
`read_environment()` y la delegación read-en-vez-de-skip para ENVSXP/S4
se mantienen deliberadamente y ahora llevan el razonamiento en comentario:
los scopes padre se registran en la tabla de referencias y un REFSXP
posterior que resolviera a un padre "saltado" devolvería contenido vacío
sin error — el coste extra es el precio de la corrección, acotado por
`max_depth`. El tope de 1024 nombres en `read_namespace_spec()` queda
comentado como límite defensivo (R escribe exactamente dos entradas:
nombre y versión). CLOSXP/LANGSXP/fórmulas siguen fuera de alcance por
diseño: reconstruir sintaxis R desde el AST serializado es arriesgado y
el fallo limpio con nombre de tipo ya permite al llamador decidir.

## Cierre de P0/P1 y apertura de la siguiente capa (2026-07-16)

La revisión de seguridad quedó cerrada con validación estricta de `REFSXP`
(incluido el índice cero), profundidad máxima simétrica en lectura y
`skip_item()`, preservación de ciclos/identidad compartida en environments,
decodificación de `CHARSXP` según flags y encoding nativo, y límites separados
para longitud lógica y bytes de asignación. Un archivo malformado produce
`InvalidRDS`; exceder una política configurada produce `RDSLimitError`, no
`UnsupportedRDS`. Pruebas de truncamiento, mutación de bits y fuzz acotado fijan
este contrato.

En pandas, las columnas simples llegan como arrays/ExtensionArrays y el
`DataFrame` se construye una sola vez, sin alineación previa de `Series`.
`Date`, `POSIXct` y `difftime` pequeños usan conversión NumPy directa. En el
archivo real `archive.rds` (27 315 tablas), el tiempo observado bajó de 35.160 s
a 20.248 s.

El prototipo Cython se mantuvo deliberadamente limitado a
`skip_string_elements`: no decodifica strings, no posee el stream y no altera
el estado del parser. En `datos_limpios.rds`, el benchmark aislado más reciente
midió 32.559 s en Python y 7.147 s con Cython (**4.56x**), verificando catálogos
idénticos. `benchmarks/benchmark_cython_skip.py` reproduce la comparación en
procesos separados.

La capa posterior ya está abierta: `read_rds_arrow()` es pública;
`to_parquet(engine="pyarrow")` funciona sin DuckDB, mientras
`engine="duckdb"` conserva la ruta column-staged de menor memoria; y la
selección de tablas por nombre crea/reutiliza automáticamente el catálogo. La
CI ejecuta mypy estricto, cobertura mínima del 80%, el fallback sin Cython,
NumPy 1.23.5/pandas 1.5.3 y los casos corruptos/fuzz.

## API analítica diferida rumbo a 1.0 (2026-07-18)

`open_rds()` expone ahora un `RDSDataset` diferido. El constructor solo valida
la ruta; `schema`, `columns`, `shape` y `tables` usan el catálogo estructural
versionado, que conserva tipo de almacenamiento R, tipo lógico, factores,
niveles y estimaciones para vectores de ancho fijo. `select()` y `table()`
componen una proyección; `collect()`, `to_arrow()`, `to_polars()` o
`to_duckdb()` son operaciones terminales. Para un único data.frame raíz,
`collect()` reutiliza `columns=` y las columnas no seleccionadas se omiten
estructuralmente.

`inspect_rds(mode="metadata")` no construye payloads de columna y devuelve
filas, columnas, compresión, schema, factores y estimaciones disponibles. No
inventa estadísticas imposibles: el tamaño real del texto/listas y los missing
requieren leer elementos. `mode="scan"` hace esa lectura de forma explícita y
añade conteo exacto de nulos, bytes de buffers y tipo Arrow.

La integración Polars (`read_rds_polars()` / `.to_polars()`) reutiliza Arrow y
no pasa por pandas. DuckDB puede recibir la proyección como relación o como una
vista registrada con `register_duckdb()`, permitiendo SQL normal sobre el
nombre elegido. Esto no se presenta como una función nativa
`read_rds('archivo')`: esa sintaxis requiere una extensión DuckDB y pushdown
del optimizador, y permanece como trabajo posterior sin bloquear la API 1.0.

La semántica lazy también se documenta sin exagerarla: `head(n)` limita el
resultado devuelto, pero todavía consume los vectores RDS seleccionados
completos. RDS es columnar-secuencial y un contenedor comprimido debe
descomprimir los bytes anteriores para alcanzar columnas tardías.
