## Febrero 23


## Enero
Generación de resultados con 20 semillas distintas sobre varios DS
Generación de estadísticas con hypothesis testing

## ML1M 04/12

Elección del mejor modelo MLP
Ejecución de BiasedMF
Análisis de 20 ejecuciones

## ML1M 11/28

Análsisi estadísticos
...

## ML1M 11/09

Error máximo
Varios tamaños de grupo

## ML1M 09/11

Modelo bien grande. Probar con datos de ML1M

## Reunión 04/11

Necesidad de test estadístico contraste hipótesis
Sacar error máximo por grupo y medio.

Siguiente paso: Aumentar complejidad de la red.
o Siguiente: Entrenamiento para grupos, sampleado

CV Cross-Validation: https://stackoverflow.com/a/48087663/932888


## Pruebas 02/11

Probar otra vez con los embedings a varias activaciones
Tener en cuenta que el item se pone siempre a 1


## Pruebas 23/10

Generando pruebas para filmtrust y animelist.

Hacer un fichero más por cada group_size. La recomendación para el grupo será la media de las recomendaciones a cada usuario. Por la no linealidad de la DNN *DEBE* dar valores distintos (¿mejores o peores?)

Verificar los resultados de 'mlp_regresion_ng2'

## Pruebas 19/10

Activaciones 1/GRP_SIZE
truncado

Para 4, 6, 8 y 10 factores

Resultados buenos.

## Pruebas 15/10

Activaciones de entrada:

- 1
- 1 normalizado
- 1 truncado
- 1 / grpsize 
- 1 / grpsize * 2
- 1 / grpsize ** 2
