"""Guion del video: una lista de escenas, cada una con sus frases.

Cada frase tiene:
  sub: texto que se muestra en el subtítulo
  say: texto que se envía al sintetizador de voz (opcional; por defecto = sub).
       Sirve para deletrear siglas o evitar símbolos que el TTS lee mal.
"""

SCENES = [
    {
        "id": "intro",
        "lines": [
            {"sub": "Dentro de tu cerebro hay unas 86 mil millones de neuronas.",
             "say": "Dentro de tu cerebro hay unas ochenta y seis mil millones de neuronas."},
            {"sub": "Y cada una depende de una pequeña proteína para mantener su forma y funcionar bien: la proteína tau."},
        ],
    },
    {
        "id": "neurona",
        "lines": [
            {"sub": "Una neurona tiene un cuerpo celular, con su núcleo, y una prolongación larga llamada axón, que puede medir desde milímetros hasta más de un metro."},
            {"sub": "Por el axón viajan constantemente mitocondrias, vesículas y proteínas, desde el cuerpo celular hasta la sinapsis, y de regreso."},
        ],
    },
    {
        "id": "microtubulos",
        "lines": [
            {"sub": "Ese transporte ocurre sobre los microtúbulos: tubos huecos formados por una proteína llamada tubulina."},
            {"sub": "Funcionan como rieles. Proteínas motoras, como la quinesina, caminan sobre ellos llevando su carga."},
        ],
    },
    {
        "id": "tau",
        "lines": [
            {"sub": "Acá entra en escena tau: una proteína asociada a los microtúbulos, codificada por el gen MAPT, y muy abundante en los axones.",
             "say": "Acá entra en escena tau: una proteína asociada a los microtúbulos, codificada por el gen, eme, a, pe, te. Y muy abundante en los axones."},
            {"sub": "Tau se une a la superficie de los microtúbulos, los estabiliza y favorece el ensamblaje de la tubulina. Así, los rieles se mantienen firmes y ordenados."},
        ],
    },
    {
        "id": "fosforilacion",
        "lines": [
            {"sub": "Su unión se regula por fosforilación: unas enzimas, las quinasas, le agregan grupos fosfato, y otras, las fosfatasas, se los quitan."},
            {"sub": "Con pocos fosfatos, tau se adhiere al microtúbulo. Con más fosfatos, se suelta."},
            {"sub": "Este equilibrio dinámico es normal: le permite a la neurona remodelar su esqueleto interno."},
        ],
    },
    {
        "id": "patologia",
        "lines": [
            {"sub": "El problema empieza cuando ese equilibrio se rompe. En varias enfermedades, tau se hiperfosforila: acumula muchos más fosfatos de lo normal."},
            {"sub": "Entonces se desprende, y los microtúbulos se vuelven inestables y se desarman."},
            {"sub": "El transporte se interrumpe, y la carga ya no llega a las sinapsis."},
        ],
    },
    {
        "id": "agregacion",
        "lines": [
            {"sub": "Libre en el citoplasma, tau cambia de forma y empieza a pegarse a otras moléculas de tau."},
            {"sub": "Primero forma pequeños agregados, los oligómeros, que se consideran especialmente tóxicos."},
            {"sub": "Después, filamentos helicoidales apareados. Y finalmente, grandes marañas dentro de la neurona: los ovillos neurofibrilares."},
        ],
    },
    {
        "id": "propagacion",
        "lines": [
            {"sub": "Las sinapsis fallan y, con el tiempo, la neurona muere."},
            {"sub": "Además, la tau mal plegada puede pasar a neuronas conectadas y hacer que la tau sana también se pliegue mal, de forma parecida a un prion."},
            {"sub": "En el Alzheimer, este avance suele seguir un recorrido predecible: empieza en la corteza entorrinal, sigue por el hipocampo, clave para la memoria, y luego se extiende por la corteza cerebral.",
             "say": "En el Alzheimer, este avance suele seguir un recorrido predecible: empieza en la corteza entorrinal. Sigue por el hipocampo, clave para la memoria. Y luego se extiende por la corteza cerebral."},
        ],
    },
    {
        "id": "tauopatias",
        "lines": [
            {"sub": "Las enfermedades con acumulación anormal de tau se llaman tauopatías."},
            {"sub": "En el Alzheimer, los ovillos de tau dentro de las neuronas aparecen junto a las placas de beta-amiloide, que se acumulan afuera.",
             "say": "En el Alzheimer, los ovillos de tau dentro de las neuronas aparecen junto a las placas de beta amiloide, que se acumulan afuera."},
            {"sub": "Otras tauopatías son la demencia frontotemporal, la parálisis supranuclear progresiva y la encefalopatía traumática crónica."},
        ],
    },
    {
        "id": "futuro",
        "lines": [
            {"sub": "Hoy se investigan anticuerpos contra tau, fármacos que frenan su agregación y terapias que reducen su producción."},
            {"sub": "Entender cómo actúa tau es clave para, algún día, frenar la neurodegeneración."},
        ],
    },
]
